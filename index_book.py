import os
import asyncio
from pathlib import Path
from app.services.rag_service import RAGService

async def index_markdown_files():
    """Index all markdown files from the docs folder."""
    
    # Path to your docs folder (adjust if needed)
    docs_path = Path("../physical-ai-book/docs")  # Update thisls
    
    if not docs_path.exists():
        print(f"Error: Docs folder not found at {docs_path}")
        return
    
    rag_service = RAGService()
    
    # Find all .md files
    md_files = list(docs_path.rglob("*.md"))
    
    if not md_files:
        print("No markdown files found!")
        return
    
    print(f"Found {len(md_files)} markdown files")
    
    doc_id = 1
    
    for md_file in md_files:
        try:
            # Read file content
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip empty files
            if len(content.strip()) < 50:
                continue
            
            # Get relative path as source
            source = str(md_file.relative_to(docs_path))
            
            # Split content into chunks (max 1000 chars per chunk)
            chunks = split_content(content, max_length=1000)
            
            for i, chunk in enumerate(chunks):
                chunk_source = f"{source} (Part {i+1})" if len(chunks) > 1 else source
                
                print(f"Indexing: {chunk_source}")
                await rag_service.add_document(
                    text=chunk,
                    source=chunk_source,
                    doc_id=doc_id
                )
                doc_id += 1
            
            print(f"✓ Indexed: {source}")
            
        except Exception as e:
            print(f"✗ Error indexing {md_file}: {e}")
    
    print(f"\n✓ Successfully indexed {doc_id - 1} document chunks!")

def split_content(text: str, max_length: int = 1000) -> list[str]:
    """Split text into chunks by paragraphs."""
    chunks = []
    current_chunk = ""
    
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

if __name__ == "__main__":
    print("Starting book indexing...")
    asyncio.run(index_markdown_files())