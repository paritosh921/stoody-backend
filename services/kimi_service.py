"""
Kimi/Moonshot AI Service

OpenAI-compatible service for question generation using Kimi K2.5 model.
Used as an alternative to OpenAI for cost-effective question generation.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional
import httpx
from openai import AsyncOpenAI

from config_async import (
    KIMI_API_KEY,
    KIMI_BASE_URL,
    KIMI_MODEL,
    USE_KIMI_FOR_QUESTIONS,
    USE_KIMI_FOR_DIAGRAMS,
    OPENAI_CONCURRENCY_LIMIT,
)

logger = logging.getLogger(__name__)


class KimiService:
    """
    Kimi/Moonshot AI service for question generation.
    Uses OpenAI-compatible API.
    """
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.model = KIMI_MODEL
        self.api_key = KIMI_API_KEY
        self.base_url = KIMI_BASE_URL
        self._http_client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        self._concurrency_semaphore = asyncio.Semaphore(OPENAI_CONCURRENCY_LIMIT)
        self._initialized = False
    
    async def _initialize_client(self):
        """Initialize the Kimi client."""
        if self.client is not None:
            return
        
        async with self._lock:
            if self.client is not None:
                return
            
            if not self.api_key:
                raise ValueError("KIMI_API_KEY not configured. Set KIMI_API_KEY or MOONSHOT_API_KEY in environment.")
            
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=200,
                    max_keepalive_connections=50,
                    keepalive_expiry=30
                ),
                timeout=httpx.Timeout(300.0)  # 5 minute timeout for large content
            )
            
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=self._http_client,
            )
            
            self._initialized = True
            logger.info(f"✅ Kimi client initialized (model: {self.model}, base_url: {self.base_url})")
    
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,  # Kimi K2.5 ONLY supports temperature=0.6
        max_tokens: int = 4000,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate chat completion using Kimi.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for generation (IGNORED - Kimi K2.5 only supports 0.6)
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to KIMI_MODEL)
            system_prompt: Optional system prompt to prepend

        Returns:
            Dict with 'success', 'response', and optionally 'error'
        """
        await self._initialize_client()

        # CRITICAL: Kimi K2.5 ONLY supports temperature=0.6
        # Override any passed temperature to avoid API errors
        temperature = 0.6

        try:
            async with self._concurrency_semaphore:
                # Prepare messages
                final_messages = []

                if system_prompt:
                    final_messages.append({
                        "role": "system",
                        "content": system_prompt
                    })

                final_messages.extend(messages)

                # Use provided model or default
                use_model = model or self.model

                logger.info(f"Kimi request: model={use_model}, messages={len(final_messages)}, temp={temperature}, max_tokens={max_tokens}")
                
                response = await self.client.chat.completions.create(
                    model=use_model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                    extra_body={"thinking": {"type": "disabled"}},  # Disable thinking for direct responses
                )
                
                # Log full response structure for debugging
                logger.info(f"Kimi raw response choices: {len(response.choices)}")
                
                if response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    message = choice.message
                    
                    # Log message attributes
                    logger.info(f"Kimi message attributes: {dir(message)}")
                    
                    # Get content - might be in different fields
                    content = message.content or ""
                    
                    # Check for reasoning_content (Kimi thinking mode)
                    reasoning_content = getattr(message, 'reasoning_content', None)
                    if reasoning_content:
                        logger.info(f"Kimi reasoning_content length: {len(reasoning_content)}")
                    
                    # If content is empty but we have reasoning, use reasoning as content
                    # This happens when kimi-k2.5 thinking mode consumes all tokens for reasoning
                    if not content and reasoning_content:
                        logger.warning("Kimi returned reasoning but no content - using reasoning_content as response")
                        content = reasoning_content
                    
                    logger.info(f"Kimi response content length: {len(content) if content else 0}")
                    
                    return {
                        "success": True,
                        "response": content,
                        "model": use_model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                            "total_tokens": response.usage.total_tokens if response.usage else 0,
                        }
                    }
                else:
                    logger.error("Kimi returned no choices in response")
                    return {
                        "success": False,
                        "error": "No choices in response",
                        "response": None,
                    }
                
        except Exception as e:
            logger.error(f"Kimi API error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "response": None,
            }
    
    async def analyze_image_async(
        self,
        image_data: str,
        prompt: str,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Analyze an image using Kimi K2.5 vision capabilities.

        Args:
            image_data: Base64 encoded image data URL (data:image/png;base64,...)
            prompt: The prompt/question about the image
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'success', 'response', and optionally 'error'
        """
        await self._initialize_client()

        try:
            async with self._concurrency_semaphore:
                # Build message with image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]

                logger.info(f"Kimi vision request: model={self.model}, prompt_length={len(prompt)}")

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.6,  # Kimi K2.5 only supports temperature=0.6
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )

                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content or ""

                    logger.info(f"Kimi vision response length: {len(content)}")

                    return {
                        "success": True,
                        "response": content,
                        "model": self.model,
                    }
                else:
                    return {
                        "success": False,
                        "error": "No choices in response",
                        "response": None,
                    }

        except Exception as e:
            logger.error(f"Kimi vision API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None,
            }

    async def chat_completion_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.6,  # Kimi K2.5 ONLY supports temperature=0.6
        max_tokens: int = 2000,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """
        Generate chat completion with tool/function calling using Kimi K2.5.

        Args:
            messages: List of message dicts
            tools: List of tool definitions (OpenAI function format)
            model: Model to use (defaults to KIMI_MODEL)
            temperature: Temperature for generation (IGNORED - Kimi K2.5 only supports 0.6)
            max_tokens: Maximum tokens in response
            tool_choice: Tool choice mode ("auto", "none", or specific tool)

        Returns:
            Dict with 'success', 'message' (with potential tool_calls), and optionally 'error'
        """
        await self._initialize_client()

        # CRITICAL: Kimi K2.5 ONLY supports temperature=0.6
        # Override any passed temperature to avoid API errors
        temperature = 0.6

        try:
            async with self._concurrency_semaphore:
                use_model = model or self.model

                logger.info(f"Kimi tool request: model={use_model}, tools={len(tools)}, messages={len(messages)}, temp={temperature}")

                response = await self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )

                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message

                    # Convert to dict format expected by callers
                    message_dict = {
                        "role": message.role,
                        "content": message.content,
                    }

                    # Include tool calls if present
                    if message.tool_calls:
                        message_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in message.tool_calls
                        ]

                    logger.info(f"Kimi tool response: tool_calls={len(message.tool_calls) if message.tool_calls else 0}")

                    return {
                        "success": True,
                        "message": message_dict,
                        "model": use_model,
                    }
                else:
                    return {
                        "success": False,
                        "error": "No choices in response",
                        "message": None,
                    }

        except Exception as e:
            logger.error(f"Kimi tool API error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "message": None,
            }

    async def health_check(self) -> Dict[str, Any]:
        """Check if Kimi service is healthy."""
        try:
            await self._initialize_client()

            # Simple test request
            response = await self.chat_completion_async(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )

            return {
                "healthy": response.get("success", False),
                "model": self.model,
                "base_url": self.base_url,
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }
    
    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            self.client = None
            self._initialized = False


# Singleton instance
_kimi_service: Optional[KimiService] = None


def get_kimi_service() -> KimiService:
    """Get or create the Kimi service singleton."""
    global _kimi_service
    if _kimi_service is None:
        _kimi_service = KimiService()
    return _kimi_service


def is_kimi_enabled() -> bool:
    """Check if Kimi is enabled for question generation."""
    return USE_KIMI_FOR_QUESTIONS and bool(KIMI_API_KEY)


def is_kimi_enabled_for_diagrams() -> bool:
    """Check if Kimi is enabled for diagram generation/verification."""
    return USE_KIMI_FOR_DIAGRAMS and bool(KIMI_API_KEY)
