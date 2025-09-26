"""Answer generator submodule for RAG pipeline."""


from typing import List, Dict, Any, Optional
import openai
from langchain_openai import ChatOpenAI  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..shared.config import settings
from ..shared.logging import setup_logging

logger = setup_logging("rag-generator")


class AnswerGenerator:
    """Answer generator using LLM with retrieved context."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize the answer generator.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = None
        self.prompt_template = None
        self.llm_chain = None
        self.model_name = model_name or settings.llm_model
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM and prompt template."""
        try:
            # Set OpenAI API key
            openai.api_key = self.api_key
            
            # Initialize LangChain ChatOpenAI (updated)
            self.llm = ChatOpenAI(
                openai_api_key=self.api_key,
                temperature=0.7,
                max_tokens=1000,
                model_name = self.model_name
            )
            
            # Create prompt template
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=self._get_disney_prompt_template()
            )
            
            # Create LLM chain
            self.llm_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template
            )
            
            logger.info("Answer generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize answer generator: {str(e)}")
            raise
    
    def _get_disney_prompt_template(self) -> str:
        """Get the Disney-specific prompt template."""
        return """You are an AI assistant helping Disney customer experience teams analyze customer reviews about Disney parks. 

Based on the following context from Disney customer reviews, please answer the user's question about Disney park experiences.

Context:
{context}

Question: {question}

Instructions:
- Provide a helpful and accurate answer based on the context provided
- Focus on insights that would be valuable for Disney customer experience teams
- If the context doesn't contain enough information to answer the question, say so
- Be specific and reference relevant details from the reviews when possible
- Maintain a professional and helpful tone

Answer:"""
    
    def generate_answer(
        self, 
        question: str, 
        context_docs: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate an answer using the LLM with retrieved context.
        
        Args:
            question: User question
            context_docs: List of relevant context documents
            temperature: LLM temperature setting
            
        Returns:
            Dictionary with generated answer and metadata
        """
        try:
            if not self.llm_chain:
                raise Exception("LLM chain not initialized")
            
            # Format context from documents
            context_text = self._format_context(context_docs)
            
            # Update LLM temperature if different
            if temperature != 0.7:
                self.llm.temperature = temperature
            
            # Generate answer
            response = self.llm_chain.run(
                context=context_text,
                question=question
            )
            
            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(context_docs, response)
            
            return {
                "answer": response.strip(),
                "confidence": confidence,
                "context_used": len(context_docs),
                "context_length": len(context_text)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while generating an answer. Please try again.",
                "confidence": 0.0,
                "context_used": 0,
                "context_length": 0
            }
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format context documents into a single text.
        
        Args:
            context_docs: List of context documents
            
        Returns:
            Formatted context text
        """
        if not context_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get("content", "")
            relevance_score = doc.get("relevance_score", 0)
            
            context_parts.append(
                f"Review {i} (Relevance: {relevance_score:.2f}):\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(
        self, 
        context_docs: List[Dict[str, Any]], 
        answer: str
    ) -> float:
        """Calculate confidence score for the generated answer.
        
        Args:
            context_docs: List of context documents used
            answer: Generated answer
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not context_docs:
                return 0.0
            
            # Base confidence on context quality
            avg_relevance = sum(
                doc.get("relevance_score", 0) for doc in context_docs
            ) / len(context_docs)
            
            # Adjust based on answer length (longer answers often more confident)
            answer_length_factor = min(len(answer) / 200, 1.0)
            
            # Combine factors
            confidence = (avg_relevance * 0.7) + (answer_length_factor * 0.3)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5  # Default moderate confidence


# Global generator instance
_generator = None


def get_generator() -> AnswerGenerator:
    """Get or create generator instance."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator
