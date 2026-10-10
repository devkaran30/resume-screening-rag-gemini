
import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
    """List of applicant IDs whose resumes should be retrieved."""

    id_list: List[str] = Field(
        ...,
        description="List of IDs of the applicants to retrieve resumes for",
    )


class JobDescription(BaseModel):
    """Job description used to retrieve similar resumes."""

    job_description: str = Field(
        ...,
        description="Job description to find suitable applicants for",
    )


class RAGRetriever:
    def __init__(self, vectorstore_db, df):
        self.vectorstore = vectorstore_db
        self.df = df

    def __reciprocal_rank_fusion__(
        self, document_rank_list: list[dict], k=50
    ):
        fused_scores = {}

        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                if doc not in fused_scores:
                    fused_scores[doc] = 0

                fused_scores[doc] += 1 / (rank + k)

        reranked_results = {
            doc: score
            for doc, score in sorted(
                fused_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        return reranked_results

    def __retrieve_docs_id__(self, question: str, k=50):
        docs_score = self.vectorstore.similarity_search_with_score(
            question, k=k
        )

        return {
            str(doc.metadata["ID"]): score
            for doc, score in docs_score
        }

    def retrieve_id_and_rerank(self, subquestion_list: list):
        document_rank_list = []

        for subquestion in subquestion_list:
            document_rank_list.append(
                self.__retrieve_docs_id__(
                    subquestion,
                    RAG_K_THRESHOLD,
                )
            )

        return self.__reciprocal_rank_fusion__(document_rank_list)

    def retrieve_documents_with_id(
        self, doc_id_with_score: dict, threshold=5
    ):
        id_resume_dict = dict(
            zip(
                self.df["ID"].astype(str),
                self.df["Resume"],
            )
        )

        retrieved_ids = sorted(
            doc_id_with_score,
            key=doc_id_with_score.get,
            reverse=True,
        )[:threshold]

        retrieved_documents = []

        for applicant_id in retrieved_ids:
            resume = id_resume_dict.get(applicant_id)

            if resume is not None:
                retrieved_documents.append(
                    f"Applicant ID {applicant_id}\n{resume}"
                )

        return retrieved_documents


class SelfQueryRetriever(RAGRetriever):
    def __init__(self, vectorstore_db, df):
        super().__init__(vectorstore_db, df)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an expert in talent acquisition.

                    Choose exactly one appropriate tool:
                    - retrieve_applicant_id when the user asks about
                      specific applicant IDs.
                    - retrieve_applicant_jd when the user asks to find
                      candidates based on skills, qualifications,
                      experience, or a job description.

                    If the user is asking a general question that does
                    not require retrieving resumes, answer directly
                    without calling a tool.
                    """,
                ),
                ("human", "{input}"),
            ]
        )

        self.meta_data = {
            "rag_mode": "",
            "query_type": "no_retrieve",
            "extracted_input": "",
            "subquestion_list": [],
            "retrieved_docs_with_scores": {},
        }

    def retrieve_docs(self, question: str, llm, rag_mode: str):

        @tool(args_schema=ApplicantID)
        def retrieve_applicant_id(id_list: List[str]):
            """Retrieve resumes for the specified applicant IDs."""

            retrieved_resumes = []

            for applicant_id in id_list:
                matches = self.df[
                    self.df["ID"].astype(str) == str(applicant_id)
                ]

                if matches.empty:
                    continue

                resume_row = matches.iloc[0]
                retrieved_resumes.append(
                    f"Applicant ID {resume_row['ID']}\n"
                    f"{resume_row['Resume']}"
                )

            return retrieved_resumes

        @tool(args_schema=JobDescription)
        def retrieve_applicant_jd(job_description: str):
            """Retrieve resumes similar to the given job description."""

            subquestion_list = [job_description]

            if rag_mode == "RAG Fusion":
                subquestion_list.extend(
                    llm.generate_subquestions(question)
                )

            self.meta_data["subquestion_list"] = subquestion_list

            retrieved_ids = self.retrieve_id_and_rerank(
                subquestion_list
            )

            self.meta_data["retrieved_docs_with_scores"] = retrieved_ids

            return self.retrieve_documents_with_id(retrieved_ids)

        self.meta_data["rag_mode"] = rag_mode
        self.meta_data["query_type"] = "no_retrieve"
        self.meta_data["extracted_input"] = ""
        self.meta_data["subquestion_list"] = []
        self.meta_data["retrieved_docs_with_scores"] = {}

        tools = [
            retrieve_applicant_id,
            retrieve_applicant_jd,
        ]

        # Gemini-compatible modern tool-calling interface.
        llm_with_tools = llm.llm.bind_tools(tools)

        response = (
            self.prompt | llm_with_tools
        ).invoke({"input": question})

        # No retrieval tool was selected.
        # Return an empty list because the caller expects a list of
        # resume documents, not a string of answer characters.
        if not response.tool_calls:
            return []

        # Execute the first selected retrieval tool.
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        tool_map = {
            "retrieve_applicant_id": retrieve_applicant_id,
            "retrieve_applicant_jd": retrieve_applicant_jd,
        }

        if tool_name not in tool_map:
            raise ValueError(
                f"Unknown retrieval tool returned by the model: {tool_name}"
            )

        self.meta_data["query_type"] = tool_name
        self.meta_data["extracted_input"] = tool_args

        return tool_map[tool_name].invoke(tool_args)
