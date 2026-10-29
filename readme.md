# Semantic Doc Search

Busca semântica em documentos (PDF/TXT) usando embeddings multilíngues e FAISS, com interface em Streamlit.

## Como funciona
1. O texto é extraído e dividido em trechos com sobreposição
2. Cada trecho vira um embedding (`paraphrase-multilingual-MiniLM-L12-v2`)
3. Os vetores são indexados no FAISS
4. A pergunta também vira embedding e retorna os trechos mais similares por cosseno

## Como rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```

![Demo](demo.png)

## Stack
Python, Streamlit, Sentence-Transformers, FAISS, pypdf

## Próximos passos
- Adicionar um LLM para gerar respostas a partir dos trechos (RAG completo)
- Persistir o índice em disco
- Deploy no Streamlit Cloud