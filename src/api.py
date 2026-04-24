import time                                             
from fastapi import FastAPI, HTTPException                                                                                                                                                                               
from pydantic import BaseModel, Field
from src.generate import generate                                                                                                                                                                                        
                                                                                                                                                                                                                           
app = FastAPI(                      
      title="Rabobank RAG Assessment API",                                                                                                                                                                                 
      description="RAG over Wikipedia (subprime mortgage crisis) via Azure OpenAI",                                                                                                                                        
      version="0.1.0",                                                                                                                                                                                                     
  )                                                                                                                                                                                                                        
                                                                                                                                                                                                                           
                                                                                                                                                                                                                           
class QueryRequest(BaseModel):      
      question: str = Field(..., min_length=3, max_length=1000)                                                                                                                                                            
                                                          
                                      
class QueryResponse(BaseModel):
      answer: str
      retrieved_chunks: list[str]
      latency_ms: float                                                                                                                                                                                                    
  
                                                                                                                                                                                                                           
@app.get("/health")                                     
def health() -> dict[str, str]:     
      return {"status": "ok"}                                                                                                                                                                                              
  
                                                                                                                                                                                                                           
@app.post("/query", response_model=QueryResponse)       
def query(req: QueryRequest) -> QueryResponse:
      start = time.perf_counter()
      try:                                                                                                                                                                                                                 
          result = generate(req.question)
      except Exception as e:                                                                                                                                                                                               
          raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
      return QueryResponse(                                                                                                                                                                                                
          question=result.question,
          answer=result.answer,                                                                                                                                                                                            
          retrieved_chunks=result.retrieved_chunks,       
          model=result.model,                                                                                                                                                                                              
          latency_ms=round((time.perf_counter() - start) * 1000, 1),                                                                                                                                                      
      )                           