import json
import httpx
from typing import List, Dict, Any, Generator, Optional


class OllamaClient:
    def __init__(self, base_url: str, default_model: str, temperature: float):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.temperature = temperature


    def is_online(self) -> bool:
        """Checks if the Ollama daemon is reachable."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False


    def list_models(self) -> List[str]:
        """Lists all downloaded models in local Ollama."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []


    def pull_model(self, model_name: str) -> Generator[Dict[str, Any], None, None]:
        """Pulls a model from Ollama library, yielding status updates."""
        url = f"{self.base_url}/api/pull"
        with httpx.stream("POST", url, json={"name": model_name}, timeout=None) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Failed to pull model {model_name}: {response.text}")
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)


    def stream_chat(
        self,
        query: str,
        book_title: str,
        retrieved_chunks: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streams chat completion tokens from Ollama."""
        chosen_model = model or self.default_model
        url = f"{self.base_url}/api/chat"

        messages = self.__build_literary_messages(
            query=query,
            book_title=book_title,
            retrieved_chunks=retrieved_chunks
        )

        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            }
        }

        with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error ({response.status_code}): {response.text}")
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        yield content


    def __build_literary_messages(
        self,
        query: str,
        book_title: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Builds a structured prompt instructing the model to provide scene context,
        chapter citations, and quote grounding.
        """
        system_prompt = (
            "You are an expert literary scholar and story analyst. "
            "Your task is to answer questions about a book based strictly on the provided excerpts.\n\n"
            "Guidelines:\n"
            "1. Identify and state the exact Chapter / Section name where the scene or event takes place as indicated in the excerpts.\n"
            "2. Provide rich context for the scene: explain who is present, the setting/location, and the dramatic context.\n"
            "3. Ground your explanation with direct quotes or textual references from the excerpts.\n"
            "4. If the provided excerpts do not contain enough information to answer the question, clearly state what is known and what is missing.\n"
            "5. Keep your tone informative, observant, and objective."
        )

        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            c_title = chunk.get("chapter_title", "Unknown")
            text = chunk.get("text", "").strip()
            context_blocks.append(
                f"--- [Excerpt {idx} | Chapter: {c_title}] ---\n{text}"
            )

        context_str = "\n\n".join(context_blocks)

        user_message = (
            f"Book: \"{book_title}\"\n\n"
            f"Context Excerpts:\n{context_str}\n\n"
            f"Question: {query}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
