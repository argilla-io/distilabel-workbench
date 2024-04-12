import streamlit as st

from hub import push_dataset_to_hub

N_PERSPECTIVES = 10
N_TOPICS = 10
N_EXAMPLES = 5

st.markdown(
    "# 🧑‍🌾 app-for-building-diverse-domain-specific-datasets-for-aligning-models"
)
st.markdown(
    "This app helps you create a dataset seed for building diverse domain-specific datasets for aligning models."
)
st.markdown(
    "Alignment datasets are used to fine-tune models to a specific domain or task, but as yet, there's a shortage of diverse datasets for this purpose."
)

project_name = st.text_input("Project Name")
domain = st.text_input("Domain")
hub_username = st.text_input("Hub Username")

st.header("Domain Perspectives")
perspectives = [
    st.text_input(f"Domain Perspective {n}") for n in range(1, N_PERSPECTIVES + 1)
]

st.header("Domain Topics")
topics = [st.text_input(f"Domain Topic {n}") for n in range(1, N_TOPICS + 1)]

st.header("Examples")

questions_answers = []

for n in range(N_EXAMPLES):
    st.subheader(f"Example {n + 1}")
    _question = st.text_area("Question", key=f"question_{n}")
    _answer = st.text_area("Answer", key=f"answer_{n}")
    questions_answers.append((_question, _answer))


if st.button("Create Dataset Seed :seedling:"):
    domain_data = {
        "perspectives": list(filter(None, perspectives)),
        "topics": list(filter(None, topics)),
        "examples": [{"question": q, "answer": a} for q, a in questions_answers],
        "domain": domain,
    }

    push_dataset_to_hub(domain_data, project_name, domain, hub_username)
