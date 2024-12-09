import gradio as gr
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
import numpy as np

hf_access_token = "HUGGINGFACE_ACCESS_TOKEN"

hf_client = InferenceClient(token=hf_access_token)

vector_db_client = QdrantClient(url="http://localhost:8000")

vector_db_name = "repo_embeddings"
author_identifier = "079da71d-e359-4988-8605-ba8a2c24f404"


def query_vector_db(user_input: str, num_results: int = 5):
    query_embedding = hf_client.feature_extraction(user_input)

    avg_embedding = query_embedding.mean(axis=1).flatten()

    reduced_vector = avg_embedding[:384]

    filter_conditions = Filter(
        must=[
            FieldCondition(
                key="author_identifier", match=MatchValue(value=str(author_identifier))
            )
        ]
    )

    # Searchdatabase
    search_results = vector_db_client.search(
        collection_name=vector_db_name,
        query_vector=reduced_vector,
        limit=num_results,
        with_payload=True,
        query_filter=filter_conditions,
    )

    retrieved_content = [result.payload["content"] for result in search_results]

    return retrieved_content


def generate_response(
    user_query,
    chat_log: list[tuple[str, str]],
    token_limit,
    temp_setting,
    nucleus_sampling,
):
    matching_docs = query_vector_db(user_query)

    conversation_history = []
    if matching_docs:
        doc_context = "\n".join(matching_docs)
        conversation_history.append(
            {
                "role": "system",
                "content": f"Here are relevant references for your query:\n{doc_context}",
            }
        )

    for turn in chat_log:
        if turn[0]:
            conversation_history.append({"role": "user", "content": turn[0]})
        if turn[1]:
            conversation_history.append({"role": "assistant", "content": turn[1]})
    conversation_history.append({"role": "user", "content": user_query})

    # make the a response
    reply = ""
    for msg in hf_client.chat_completion(
        conversation_history,
        max_tokens=token_limit,
        stream=True,
        temperature=temp_setting,
        top_p=nucleus_sampling,
    ):
        content_chunk = msg.choices[0].delta.content
        reply += content_chunk

    return reply


sample_questions = [
    "What are the key steps for pathfinding in robotics?",
    "Provide sample code for navigating to a waypoint.",
    "How do I integrate ROS 2 with navigation algorithms?",
]

with gr.Blocks() as chat_interface:
    bot_ui = gr.Chatbot()

    with gr.Row():
        user_input_box = gr.Textbox(label="Enter your message")

    question_selector = gr.Dropdown(
        choices=[""] + sample_questions, label="Example Queries"
    )

    with gr.Accordion("Advanced Settings", open=False):
        token_slider = gr.Slider(
            minimum=50, maximum=2048, value=512, step=50, label="Max Tokens"
        )
        temp_slider = gr.Slider(
            minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"
        )
        nucleus_slider = gr.Slider(
            minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p Sampling"
        )

    # chat history
    chat_memory = gr.State([])

    def handle_user_input(user_text, chat_log, token_max, temp_val, top_p_val):
        if chat_log is None:
            chat_log = []
        assistant_reply = generate_response(
            user_text, chat_log, token_max, temp_val, top_p_val
        )
        chat_log.append((user_text, assistant_reply))
        return chat_log, chat_log

    def handle_question_selection(
        example_question, chat_log, token_max, temp_val, top_p_val
    ):
        if not example_question:
            return chat_log, chat_log
        return handle_user_input(
            example_question, chat_log, token_max, temp_val, top_p_val
        )

    user_input_box.submit(
        handle_user_input,
        inputs=[user_input_box, chat_memory, token_slider, temp_slider, nucleus_slider],
        outputs=[bot_ui, chat_memory],
    )

    question_selector.change(
        handle_question_selection,
        inputs=[
            question_selector,
            chat_memory,
            token_slider,
            temp_slider,
            nucleus_slider,
        ],
        outputs=[bot_ui, chat_memory],
    )

if __name__ == "__main__":
    chat_interface.launch(server_name="0.0.0.0", server_port=3001)
