from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

text = """Modern technology continues to reshape the way people communicate and access information across the globe. Innovations like artificial intelligence and cloud computing are making systems faster, smarter, and more efficient. Psychology helps us understand human behavior, thoughts, and emotions in different situations. Studying psychological patterns can improve relationships and enhance personal well-being.

Nature offers breathtaking landscapes that inspire peace and admiration in those who experience them. From towering mountains to calm oceans, every element of nature has its own unique charm. The changing seasons paint the world in different colors, adding to its endless beauty. Spending time in nature can refresh the mind and create a deep sense of connection with the environment."""

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

result = splitter.split_text(text)

print(f"Splits Made - {len(result)}")
print()
count = 1
for i in result:
    print(f"{count}. {i}")
    print()
    count = count + 1


