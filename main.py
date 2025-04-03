import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import tempfile

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_chatbot.log")
    ]
)
logger = logging.getLogger("PDFChatbot")


class PDFChatbot:
    """PDF dosyalarından bilgi çıkarıp sorgulamaya olanak tanıyan chatbot sınıfı."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_dir: str = "./chroma_db"):
        """
        PDF chatbot başlatma

        Args:
            model_name: Kullanılacak LLM modeli
            embedding_model_name: Kullanılacak embedding modeli
            persist_dir: Vektör depolama klasörü
        """
        self.documents = []
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.persist_dir = persist_dir

        # Embedding modelini yükle
        logger.info("Embedding modeli yükleniyor...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Metin ayırıcı oluştur
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # LLM modelini yükle
        logger.info("Dil modeli yükleniyor...")
        self.setup_model()

        # Eğer mevcut bir vektör depo varsa yükle
        self._load_existing_vectorstore()

    def _load_existing_vectorstore(self) -> None:
        """Eğer varsa, mevcut vektör deposunu yükle."""
        try:
            if os.path.exists(self.persist_dir):
                logger.info(f"Mevcut vektör depo bulundu: {self.persist_dir}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                # Retriever oluştur
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

                # QA zinciri oluştur
                if hasattr(self, 'llm'):
                    self._setup_qa_chain()
                    logger.info("Mevcut vektör depo başarıyla yüklendi")
        except Exception as e:
            logger.warning(f"Mevcut vektör depo yüklenirken hata: {e}")

    def setup_model(self) -> bool:
        """LLM modelini kurar."""
        try:
            # API token kontrolü
            if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
                logger.error("HUGGINGFACEHUB_API_TOKEN çevre değişkeni ayarlanmamış!")
                return False

            # HuggingFace Hub üzerinden modeli yükle
            self.llm = HuggingFaceHub(
                repo_id=self.model_name,
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )
            logger.info("Model başarıyla yüklendi.")
            return True
        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu: {e}")
            return False

    def _setup_qa_chain(self) -> None:
        """Soru-Cevap zincirini oluşturur."""
        if not self.retriever:
            logger.warning("Retriever oluşturulmadan QA zinciri kurulamaz.")
            return

        logger.info("Soru-Cevap zinciri hazırlanıyor...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def process_pdf(self, pdf_path: Union[str, Path]) -> bool:
        """
        PDF dosyasını işler ve vektör depoya kaydeder.

        Args:
            pdf_path: İşlenecek PDF dosyasının yolu

        Returns:
            bool: İşlem başarılı oldu mu
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF dosyası bulunamadı: {pdf_path}")
                return False

            logger.info(f"PDF işleniyor: {pdf_path}")

            # PDF'i yükle ve belgeleri çıkar
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            logger.info(f"PDF'den {len(documents)} sayfa yüklendi.")

            # Belgeleri parçalara ayır
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Belgeler {len(split_docs)} parçaya ayrıldı.")
            self.documents.extend(split_docs)

            # Vektör depoyu oluştur veya güncelle
            if self.vector_store is None:
                logger.info("Vektör depo oluşturuluyor...")
                self.vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir
                )
                self.vector_store.persist()
            else:
                logger.info("Vektör depo güncelleniyor...")
                self.vector_store.add_documents(split_docs)
                self.vector_store.persist()

            # Retriever oluştur
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            # QA zinciri oluştur
            self._setup_qa_chain()

            logger.info(f"PDF başarıyla işlendi ve indekslendi: {pdf_path.name}")
            return True

        except Exception as e:
            logger.error(f"PDF işlenirken bir hata oluştu: {e}", exc_info=True)
            return False

    def ask_question(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Vektör depoyu kullanarak sorguyu yanıtlar.

        Args:
            question: Kullanıcı sorusu

        Returns:
            Optional[Dict]: Yanıt ve kaynak belgelerle birlikte sonuç sözlüğü
        """
        if self.qa_chain is None:
            logger.warning("Soru sorulmadan önce bir PDF dosyası yüklenmelidir.")
            return {"result": "Lütfen önce bir PDF dosyası yükleyin.", "source_documents": []}

        try:
            logger.info(f"Soru işleniyor: {question}")
            result = self.qa_chain({"query": question})

            # Yanıtı temizle ve yapılandır
            clean_result = result["result"].encode('utf-8', errors='ignore').decode('utf-8')

            logger.info(f"Yanıt oluşturuldu ({len(clean_result)} karakter)")
            logger.debug(f"Yanıt içeriği: {clean_result[:100]}...")

            return {
                "result": clean_result,
                "source_documents": result["source_documents"],
                "query": question
            }

        except Exception as e:
            logger.error(f"Soru yanıtlanırken bir hata oluştu: {e}", exc_info=True)
            return {
                "result": f"Üzgünüm, sorunuzu yanıtlarken bir hata oluştu: {str(e)}",
                "source_documents": [],
                "query": question
            }

    def get_document_count(self) -> int:
        """İşlenen toplam döküman parçası sayısını döndürür."""
        return len(self.documents)

    def clear_documents(self) -> None:
        """Tüm belge verilerini temizler."""
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
                self.vector_store = None
            self.documents = []
            self.retriever = None
            self.qa_chain = None
            logger.info("Tüm belgeler ve vektör depo temizlendi.")
        except Exception as e:
            logger.error(f"Belgeler temizlenirken hata oluştu: {e}")


def main():
    """Ana uygulama fonksiyonu."""
    print("\n===== PDF Tabanlı Soru-Cevap Sistemi =====")
    print("Bu uygulama PDF dosyalarından bilgi çıkararak sorularınızı yanıtlar.")

    # API token kontrolü
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        api_token = input("HuggingFace API token'ınızı girin: ")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

    # Chatbot örneğini oluştur
    chatbot = PDFChatbot()

    while True:
        print("\n=== ANA MENÜ ===")
        print("1. PDF Dosyası veya Klasörü İşle")
        print("2. Soru Sor")
        print("3. Belge İstatistikleri")
        print("4. Belgeleri Temizle")
        print("5. Çıkış")

        choice = input("\nSeçiminiz (1-5): ")

        if choice == "1":
            # PDF işleme
            pdf_path = input("PDF dosyası veya klasör yolu girin: ")
            processed = False

            if os.path.isdir(pdf_path):
                # Dizindeki PDF'leri işle
                for filename in os.listdir(pdf_path):
                    if filename.lower().endswith('.pdf'):
                        file_path = os.path.join(pdf_path, filename)
                        print(f"\nDosya işleniyor: {filename}")
                        success = chatbot.process_pdf(file_path)
                        if success:
                            processed = True
                            print(f"✅ {filename} başarıyla işlendi!")
                        else:
                            print(f"❌ {filename} işlenirken hata oluştu.")

            elif os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
                # Tek dosya işle
                success = chatbot.process_pdf(pdf_path)
                if success:
                    processed = True
                    print(f"✅ {os.path.basename(pdf_path)} başarıyla işlendi!")
                else:
                    print(f"❌ {os.path.basename(pdf_path)} işlenirken hata oluştu.")
            else:
                print("⚠️ Geçerli bir PDF dosyası veya dizini belirtilmedi.")

            if not processed:
                print("Hiçbir PDF dosyası işlenemedi.")

        elif choice == "2":
            # Soru sorma
            if chatbot.qa_chain is None:
                print("⚠️ Henüz bir PDF dosyası işlenmemiş! Lütfen önce seçenek 1'i kullanın.")
                continue

            print("\n==== PDF Sorgulama Modu ====")
            print("Çıkmak için 'q' veya 'quit' yazın.")

            while True:
                question = input("\nSorunuz: ")
                if question.lower() in ['q', 'quit', 'exit', 'ana menü', 'menu']:
                    break

                if not question.strip():
                    continue

                result = chatbot.ask_question(question)

                print(f"\n🤖 Yanıt:\n{result['result']}")

                show_sources = input("\nKaynak belgeleri görmek ister misiniz? (e/h): ").lower()
                if show_sources in ['e', 'evet', 'y', 'yes']:
                    print("\n📚 Kaynak Belge Parçaları:")
                    for i, doc in enumerate(result["source_documents"]):
                        print(f"\n📄 Kaynak {i + 1}:")
                        print(f"{doc.page_content}\n")
                        print(f"Metadata: {doc.metadata}")

        elif choice == "3":
            # İstatistikler
            doc_count = chatbot.get_document_count()
            print(f"\n📊 Toplam İşlenen Belge Parçası: {doc_count}")

        elif choice == "4":
            # Belgeleri temizle
            confirm = input("Tüm belgeleri temizlemek istediğinize emin misiniz? (e/h): ")
            if confirm.lower() in ['e', 'evet', 'y', 'yes']:
                chatbot.clear_documents()
                print("🧹 Tüm belgeler temizlendi.")

        elif choice == "5":
            # Çıkış
            print("\nUygulamadan çıkılıyor. Görüşmek üzere!")
            break

        else:
            print("❌ Geçersiz seçim. Lütfen 1-5 arasında bir sayı girin.")


if __name__ == "__main__":
    main()