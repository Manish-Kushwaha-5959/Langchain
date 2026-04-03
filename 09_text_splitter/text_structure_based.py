from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """The quick brown fox jumps over the lazy dog and runs into the forest to find its friends.
This text, commonly used as a placeholder, helps in evaluating how a text splitter handles various scenarios like punctuation, spaces, and paragraph breaks.
Text splitters are crucial tools in natural language processing (NLP) for breaking down large documents into manageable chunks.
In the realm of digital design and development, filler text like Lorem Ipsum is widely used for previewing layouts without distracting the viewer with meaningful content.

You can generate various amounts of such text using online generators or built-in functions in some software like Microsoft Word.
Some generators use real books to create text that looks natural but has no actual meaning.
Different splitting methods exist, such as splitting by a specific character, a regular expression, a fixed length, or even a specific number of chunks.
Recursive character text splitters, for example, carefully break down text to ensure the resulting pieces maintain some level of context or coherence."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0
)

result = splitter.split_text(text)
print(len(result))
for i in result:
    print(i)


# loader = PyPDFLoader(r'09_text_splitter/documents/demo.pdf')

# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=50,
#     chunk_overlap=0
# )

# result = splitter.split_documents(docs)

# print(len(docs))
# print(len(result))
# print(result[0].page_content)
# print(result[1].page_content)

