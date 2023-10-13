import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_most_similar_response(df, query, top_k=1):
    # Step 1: Prepare Data
    vectorizer = TfidfVectorizer()
    all_data = list(df['Query']) + [query]

    # Step 2: TF-IDF Vectorizations
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Step 3: Compute Similarity
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)

    # Step 4: Sort and Pick Top k Responses
    sorted_indexes = similarity_scores.argsort()[0][-top_k:]
    
    # Fetch the corresponding responses from the DataFrame
    most_similar_responses = df.iloc[sorted_indexes]['Response'].values
    return most_similar_responses

# Sample DataFrame with user_chat and response columns
data = {
    'Query': [
    'Hi',
    'What is the meaning of life?',
    'Is reality an illusion?',
    'What is the nature of consciousness?',
    'Can we prove the existence of God?',
    'What is the mind-body problem?',
    'Are there objective moral values?',
    'What is the nature of time?',
    'Can we truly know anything?',
    'What is the concept of justice?',
    'What is the purpose of human existence?',
    'What is the nature of beauty and aesthetics?',
    'Is free will compatible with determinism?',
    'What is the concept of truth?',
    'What is the source of human knowledge?',
    'What is the nature of the self?',
    'Is there an afterlife?',
    'What is the nature of good and evil?',
    'Is reality an illusion?',
    'What is the nature of consciousness?',
    'Can we prove the existence of God?',
    'What is the mind-body problem?',
    'Are there objective moral values?',
    'What is the nature of time?',
    'Can we truly know anything?',
    'What is the concept of justice?',
    'What is the purpose of human existence?',
    'What is the nature of beauty and aesthetics?',
    'Is free will compatible with determinism?',
    'What is the concept of truth?',
    'What is the source of human knowledge?',
    'What is the nature of the self?',
    'Is there an afterlife?',
    'What is the nature of good and evil?',
    'Is knowledge gained through experience or reason?',
    'What is the role of reason in ethics?',
    'Are humans inherently good or evil?',
    'What is the role of suffering in life?',
    'What is the relationship between science and philosophy?',
    'What is the concept of justice?',
    'What is the role of skepticism in philosophy?',
    'What is the concept of the self in Eastern philosophy?',
    'What is the relationship between ethics and religion?',
    'What is the nature of language and thought in animals?',
    'What is the role of art in addressing social and political injustices?',
    'What is the concept of the sublime in aesthetics?',
    'What is the concept of utopia and its philosophical challenges?',
    'What is the relationship between ethics and global issues?',
    'What is the nature of knowledge in the digital age?',
    'What is the concept of truth in the age of "fake news"?',
    'What is the relationship between ethics and technology ethics?',
],
'Response': [
    'Hello, im Plato the Philosophy Bot, Ask me anything!',
    'The meaning of life is a deeply philosophical question with no single, universally accepted answer. Philosophers have offered various interpretations, including pursuing personal happiness or fulfilling one\'s potential.',
    'It\'s a question that has puzzled humanity for centuries.',
    'The question of whether reality is an illusion or not is a complex one. Various philosophical traditions, including idealism and solipsism, offer different perspectives on the nature of reality.',
    'The nature of consciousness is a central question in philosophy. Some theories suggest that consciousness arises from physical processes in the brain, while others propose a more elusive, non-physical quality of mind.',
    'The existence of God is a topic of theological and philosophical debate. Arguments for and against God\'s existence, such as the cosmological argument and the problem of evil, have been explored for centuries.',
    'The mind-body problem concerns the relationship between the mind (consciousness) and the physical body. Philosophers explore whether the mind is a distinct substance or an emergent property of the body.',
    'The debate over objective moral values involves the question of whether moral principles exist independently of human beliefs and opinions. Some philosophers argue for their existence, while others contend that morals are subjective and culturally relative.',
    'Philosophers have debated the nature of time, considering whether it\'s an objective, flowing reality or an illusion. The theory of relativity in physics has added complexity to this philosophical question.',
    'Epistemology explores the nature of knowledge and asks whether we can have genuine, justified beliefs about the world. Philosophers grapple with skepticism, foundationalism, and the limits of human understanding in addressing this question.',
    'The concept of justice involves questions about fairness, ethics, and the distribution of resources or opportunities. Different philosophical traditions, such as utilitarianism, deontology, and virtue ethics, offer diverse perspectives on what constitutes justice.',
    'The question of life\'s purpose varies among philosophical and religious traditions. Some argue that life has inherent meaning, while others contend that individuals must create their own purpose.',
    'Aesthetics explores the philosophy of beauty, art, and the nature of aesthetic experiences. Philosophers investigate whether beauty is objective, subjective, or culturally constructed.',
    'The debate on free will and determinism examines whether individuals can make choices independently of causal factors. Some philosophers argue that free will is compatible with determinism, while others assert that it is not.',
    'The nature of truth is a foundational issue in philosophy. Some argue that truth is objective and absolute, while others propose that it is context-dependent and relative to perspectives.',
    'Epistemology explores the sources of knowledge, considering issues like justified belief, truth, and the reliability of our cognitive faculties in the pursuit of knowledge.',
    'Philosophers discuss the nature of personal and social identity, exploring how identity is defined, constructed, and maintained.',
    'The question of an afterlife touches on issues of mortality, the soul, and the possibility of existence beyond death. Various religious and philosophical traditions offer different answers.',
    'The nature of good and evil is a foundational question in ethics. Philosophers explore the concepts of moral virtues, vices, and the principles that govern human behavior.',
    'The question of whether reality is an illusion or not is a complex one. Various philosophical traditions, including idealism and solipsism, offer different perspectives on the nature of reality.',
    'The nature of consciousness is a central question in philosophy. Some theories suggest that consciousness arises from physical processes in the brain, while others propose a more elusive, non-physical quality of mind.',
    'The existence of God is a topic of theological and philosophical debate. Arguments for and against God\'s existence, such as the cosmological argument and the problem of evil, have been explored for centuries.',
    'The mind-body problem concerns the relationship between the mind (consciousness) and the physical body. Philosophers explore whether the mind is a distinct substance or an emergent property of the body.',
    'The debate over objective moral values involves the question of whether moral principles exist independently of human beliefs and opinions. Some philosophers argue for their existence, while others contend that morals are subjective and culturally relative.',
    'Epistemology explores the nature of knowledge and asks whether we can have genuine, justified beliefs about the world. Philosophers grapple with skepticism, foundationalism, and the limits of human understanding in addressing this question.',
    'The concept of justice involves questions about fairness, ethics, and the distribution of resources or opportunities. Different philosophical traditions, such as utilitarianism, deontology, and virtue ethics, offer diverse perspectives on what constitutes justice.',
    'The question of life\'s purpose varies among philosophical and religious traditions. Some argue that life has inherent meaning, while others contend that individuals must create their own purpose.',
    'Aesthetics explores the philosophy of beauty, art, and the nature of aesthetic experiences. Philosophers investigate whether beauty is objective, subjective, or culturally constructed.',
    'The debate on free will and determinism examines whether individuals can make choices independently of causal factors. Some philosophers argue that free will is compatible with determinism, while others assert that it is not.',
    'The nature of truth is a foundational issue in philosophy. Some argue that truth is objective and absolute, while others propose that it is context-dependent and relative to perspectives.',
    'Epistemology explores the sources of knowledge, considering issues like justified belief, truth, and the reliability of our cognitive faculties in the pursuit of knowledge.',
    'Philosophers discuss the nature of personal and social identity, exploring how identity is defined, constructed, and maintained.',
    'The question of an afterlife touches on issues of mortality, the soul, and the possibility of existence beyond death. Various religious and philosophical traditions offer different answers.',
    'The nature of good and evil is a foundational question in ethics. Philosophers explore the concepts of moral virtues, vices, and the principles that govern human behavior.',
    'Is knowledge gained through experience or reason? Epistemology examines whether knowledge primarily arises from sensory experience or innate reasoning abilities.',
    'What is the role of reason in ethics? Ethical theories often discuss the role of reason in determining what is right or good, exploring rationalism, moral reasoning, and ethical decision-making processes.',
    'Are humans inherently good or evil? Philosophers and psychologists have long debated whether humans are born with inherently good or evil qualities, or if morality is shaped by upbringing and environment.',
    'The nature of suffering is a foundational question in ethics and theodicy. Philosophers and theologians reflect on the existence of suffering and its implications for understanding the nature of life and the problem of evil.',
    'The relationship between science and philosophy has evolved over time. Philosophers explore how science informs philosophical inquiry and vice versa.',
    'The concept of justice involves questions about fairness, ethics, and the distribution of resources or opportunities. Different philosophical traditions, such as utilitarianism, deontology, and virtue ethics, offer diverse perspectives on what constitutes justice.',
    'Skepticism in philosophy questions the reliability and limits of human knowledge, prompting critical examination of beliefs and claims.',
    'Eastern philosophies, such as Buddhism and Taoism, offer unique perspectives on the nature of the self and its relationship to the world.',
    'The connection between ethics and religion is a complex one. Philosophers discuss the ethical principles that guide human behavior and examine the implications of divine command theory and natural law ethics.',
    'Philosophers explore the cognitive abilities and language-like communication in non-human animals, challenging the boundary between human and animal cognition.',
    'Aesthetics examines how art can influence social and political movements, provoke change, or challenge the status quo.',
    'The concept of the sublime involves experiences of awe, beauty, and vastness in art and nature. It has been a subject of philosophical exploration in aesthetics.',
    'The concept of utopia involves the idea of an ideal, perfect society. Philosophers examine the challenges of realizing such a society and the implications for human nature and ethics.',
    'Global ethics addresses moral questions related to international relations, global poverty, human rights, and environmental concerns.',
    'Epistemology explores how the digital age has transformed the acquisition, distribution, and reliability of knowledge, as well as the challenges it poses.',
    'The era of "fake news" challenges traditional notions of truth and raises questions about the reliability of information and the role of expertise.',
    'Technology ethics delves into the moral implications of emerging technologies, such as AI, biotechnology, and surveillance, and their impact on society.', 
],
}

df = pd.DataFrame(data)
st.title("Philosophy Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    responses = get_most_similar_response(df, prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        for response in responses:
            st.markdown(f"Plato: {response}")

    # Add assistant response to chat history
    for response in responses:
        st.session_state.messages.append({"role": "assistant", "content": f"Plato: {response}"})



#python -m venv nlp-venv
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUse
#cd .\nlp-venv\
#.\Scripts\activate

#Install extension
#pip install streamlit
#pip install pandas
#pip install scikit-learn

#To run the app
#streamlit run ..\apps\app.py
