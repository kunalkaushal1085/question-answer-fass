# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework import status
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import os
# import pdfkit
# from io import BytesIO
# from django.http import HttpResponse
# # from weasyprint import default_url_fetcher, HTML
# from langchain_community.document_loaders import UnstructuredExcelLoader
# from django.core.files.uploadedfile import InMemoryUploadedFile
# from tempfile import NamedTemporaryFile
# from django.shortcuts import render
# from dotenv import load_dotenv
# import os
# from django.conf import settings
# import  uuid
# import  pandas as pd

# # Load .env file
# load_dotenv()

# # Retrieve the API key
# # google_api_key = os.getenv("GOOGLE_API_KEY")
# # google_api_key = "AIzaSyB95BhiROhbWMrzOuDLTfXXeHA9_NpQ4PI"
# # google_api_key = "AIzaSyBEiuPyIdSMp73xU51ej9dpYsZW26nGbfk"
# google_api_key = "AIzaSyCY3LXF5fyiI6AzVIXhlWjXtlgAX1PN5Fk"
# print(google_api_key,"key++++++++++++")
# def index(request):
#     return render(request, 'abc.html')



# class UploadPDFView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request):
#         try:
#             excel_file = request.FILES.get('files')
#             if not excel_file:
#                 return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
            
#             # Save the uploaded file temporarily using NamedTemporaryFile
#             with NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
#                 for chunk in excel_file.chunks():
#                     temp_file.write(chunk)
#                 temp_file_path = temp_file.name
            
#             # Load data from the Excel file
#             docs = self.load_excel_data(temp_file_path)
            
#             # Clean up the temporary file
#             os.remove(temp_file_path)
#             self.get_vector_store(docs)
#             return Response({"message": "PDF files processed successfully."}, status=status.HTTP_200_OK)
#         except Exception as e:
#             print("inside exception:", e)
#             return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

#     def load_excel_data(self, excel_file_path: str):
#         loader = UnstructuredExcelLoader(excel_file_path, mode="elements")
#         docs = loader.load()
#         return docs
        

#     def get_pdf_text(self, pdf_docs):
#         text = ""
#         for pdf in pdf_docs:
#             pdf_reader = PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         return text

#     def get_text_chunks(self, text):
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#         chunks = text_splitter.split_text(text)
#         return chunks

#     def get_vector_store(self, text_chunks):
#         text_chunks = [doc.page_content for doc in text_chunks]

#         # Ensure the texts are not empty
#         text_chunks = [text for text in text_chunks if text]
#         # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
#         # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyBEiuPyIdSMp73xU51ej9dpYsZW26nGbfk")
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyCY3LXF5fyiI6AzVIXhlWjXtlgAX1PN5Fk")
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local("faiss_index")

# class AnswerQuestionView(APIView):
#     def post(self, request,):
#         try:
#             question_file = request.FILES.get('file')
#             print('question_file--->',question_file)
#             if not question_file:
#                 return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
            
#             questions = self.load_questions_from_excel(question_file)
#             if not questions:
#                 return Response({"error": "No questions found in the uploaded file."}, status=status.HTTP_400_BAD_REQUEST)

#             answers = []
#             for question in questions:
#                 print(question,"question")
#                 response = self.user_input(question)
#                 answers.append(response["output_text"])

#             responses_dir = os.path.join(settings.MEDIA_ROOT, 'responses')
#             os.makedirs(responses_dir, exist_ok=True)
#             output_filename = "responses_{}.xlsx".format(uuid.uuid4().hex)
#             output_path = os.path.join(settings.MEDIA_ROOT, 'responses', output_filename)
#             self.save_answers_to_excel(questions, answers, output_path)

#             file_url = request.build_absolute_uri(settings.MEDIA_URL + 'responses/' + output_filename)
#             return Response({"message": "Answers generated and saved successfully.", "file_url": file_url}, status=status.HTTP_200_OK)
#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

#     def load_questions_from_excel(self, file):
#         df = pd.read_excel(file)
#         # Attempt to find the correct column
#         possible_column_names = ['Questions', 'questions', 'Question', 'question']
#         question_column = None
#         for col in possible_column_names:
#             if col in df.columns:
#                 question_column = col
#                 break

#         if question_column is None:
#             raise ValueError("No 'Questions' column found in the uploaded file.")
        
#         return df[question_column].dropna().tolist()

#     def user_input(self, user_question):
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",max_tokens=500)
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(user_question)
#         chain = self.get_conversational_chain()
#         response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#         return response

#     def get_conversational_chain(self):
#         prompt_template = """
#         Answer the question as detailed as possible from the provided context by the user, make sure to provide all the details. If the answer is not in
#         provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.Read the question accordingly.

#         \n\nContext:\n {context}?\n
#         Question: \n{question}\n
#         Answer:
#         """
#         model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#         prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#         chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#         return chain

#     def save_answers_to_excel(self, questions, answers, output_path):
#         df = pd.DataFrame({'Questions': questions, 'Answers': answers})
#         df.to_excel(output_path, index=False)








from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import openai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from django.conf import settings
import os
import pdfkit
from io import BytesIO
from django.http import HttpResponse
from langchain_community.document_loaders import UnstructuredExcelLoader
from django.core.files.uploadedfile import InMemoryUploadedFile
from tempfile import NamedTemporaryFile
from django.shortcuts import render
from dotenv import load_dotenv
import uuid
import pandas as pd

# Load .env file
load_dotenv()

# Retrieve the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key: {openai_api_key}")
openai.api_key = openai_api_key

def index(request):
    return render(request, 'abc.html')

class UploadPDFView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        try:
            excel_file = request.FILES.get('files')
            if not excel_file:
                return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Save the uploaded file temporarily using NamedTemporaryFile
            with NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                for chunk in excel_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            # Load data from the Excel file
            docs = self.load_excel_data(temp_file_path)
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            self.get_vector_store(docs)
            return Response({"message": "PDF files processed successfully."}, status=status.HTTP_200_OK)
        except Exception as e:
            print("inside exception:", e)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def load_excel_data(self, excel_file_path: str):
        loader = UnstructuredExcelLoader(excel_file_path, mode="elements")
        docs = loader.load()
        return docs

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        text_chunks = [doc.page_content for doc in text_chunks]

        # Ensure the texts are not empty
        text_chunks = [text for text in text_chunks if text]
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o", temperature=0.3)
import re
from langchain_core.messages.ai import AIMessage
import time
class AnswerQuestionView(APIView):
    def post(self, request,):
        try:
            question_file = request.FILES.get('file')
            print(question_file,"question_file")
            if not question_file:
                return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
            
            questions = self.load_questions_from_excel(question_file)
            if not questions:
                return Response({"error": "No questions found in the uploaded file."}, status=status.HTTP_400_BAD_REQUEST)

            answers = []
            for question in questions:
                
                response = self.user_input(question)
                print('response---->',type(response))
                # answers.append(response["answer"])
                print("response[0]--->",response[0])
                answers.append(response[0]['answer'])
                print(answers,"answere")
                # time.sleep(60)
            responses_dir = os.path.join(settings.MEDIA_ROOT, 'responses')
            os.makedirs(responses_dir, exist_ok=True)
            output_filename = "responses_{}.xlsx".format(uuid.uuid4().hex)
            output_path = os.path.join(settings.MEDIA_ROOT, 'responses', output_filename)
            print(output_path,"output_path")
            self.save_answers_to_excel(questions, answers, output_path)

            file_url = request.build_absolute_uri(settings.MEDIA_URL + 'responses/' + output_filename)
            return Response({"message": "Answers generated and saved successfully.", "file_url": file_url}, status=status.HTTP_200_OK)
        except Exception as e:
            print('e--->',str(e))
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def load_questions_from_excel(self, file):
        df = pd.read_excel(file)
        # Attempt to find the correct column
        possible_column_names = ['Questions', 'questions', 'Question', 'question']
        question_column = None
        for col in possible_column_names:
            if col in df.columns:
                question_column = col
                break

        if question_column is None:
            raise ValueError("No 'Questions' column found in the uploaded file.")
        
        return df[question_column].dropna().tolist()

    def user_input(self, user_question):
        print("inside get user input")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        chain = self.get_conversational_chain(docs,user_question)
        # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print('chain---->',chain)
        return chain

    # def get_conversational_chain(self):
    #     prompt_template = """
    #     Answer the question as detailed as possible from the provided context by the user, make sure to provide all the details. If the answer is not in
    #     provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.Read the question accordingly.

    #     \n\nContext:\n {context}?\n
    #     Question: \n{question}\n
    #     Answer:
    #     """
    #     model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    #     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    #     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        # return chain
    
    # def get_conversational_chain(self, docs, question):
    #     print("inside get conversation")
    #     # Manually create the context from the docs and generate the answer using the model
    #     context = "\n".join([doc.page_content for doc in docs])
    #     print(context,"context++++++++")
    #     results = []
    #     import json
    #     prompt = f"""
    #     Answer the question as detailed as possible from the provided context by the user, make sure to provide all the details. If the answer is not in
    #     provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.

    #     Context: {context}

    #     Question: {question}

    #     Answer:
    #     """
    #     response = model.invoke(prompt)  # Generate the answer
    #     print(type(response),"response")
    #     result = {
    #     "question": question,
    #     "answer": "The answer is not available in the context."  # Default answer if response is empty
    # }
    
    #     # Check if the response is an instance of AIMessage
    #     if isinstance(response, AIMessage):
    #         # Access the content property of AIMessage
    #         if hasattr(response, 'content'):
    #             result["answer"] = response.content.strip()
    #         else:
    #             print("AIMessage does not have 'content' attribute.")
    #     elif isinstance(response, str):
    #         # If response is a string, assume it's the content directly
    #         result["answer"] = response.strip()
    #     else:
    #         print("Unexpected response type:", type(response))
        
    #     # Print the result in JSON format
    #     print(json.dumps([result], indent=4))
        
    #     return json.dumps([result], indent=4)  # R
    
    def get_conversational_chain(self, docs, questions):
        import json
        print("inside get_conversational_chain")
        
        # Create the context from the docs
        context = "\n".join([doc.page_content for doc in docs])
        # print(context, "context++++++++")
        
        results = []  # List to store all question-answer pairs

        # for question in questions:

        print('question before  prmpt---->',questions)
        # Create the prompt for each question
        prompt = f"""
        Answer the question as detailed as possible from the provided context by the user. Make sure to provide all the details. If the answer is not in
        the provided context, just say, "The answer is not available in the context." Don't provide a wrong answer.

        Context: {context}

        Question: {questions}

        Answer:
        """
        
        # Generate the answer using the model
        response = model.invoke(prompt)
        # print(dir(response), "response")
        
        # Default result structure
        result = {
            "question": questions,
            "answer": "The answer is not available in the context."
        }
        
        # Check if the response is an instance of AIMessage
        if isinstance(response, AIMessage):
            if hasattr(response, 'content'):
                # print('response.content.strip()---->',response.content.strip())
                result["answer"] = response.content.strip()
            else:
                print("AIMessage does not have 'content' attribute.")
        elif isinstance(response, str):
            result["answer"] = response.strip()
        else:
            print("Unexpected response type:", type(response))
        
        # Append the result to the list
        results.append(result)
        
        # Convert the results list to JSON and print
        # json_result = json.dumps(results, indent=4)
        print('jsonrsult----->>>>>',results)
        
        return results 
        
    
    def save_answers_to_excel(self, questions, answers, output_path):
        df = pd.DataFrame({'Questions': questions, 'Answers': answers})
        df.to_excel(output_path, index=False)

