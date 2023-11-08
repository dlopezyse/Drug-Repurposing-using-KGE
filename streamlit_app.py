#                   DRUG REPURPOSING USING KNOWLEDGE GRAPHS

##########
#LIBRARIES
##########

import streamlit as st
import pandas as pd

#############
#PAGE SET UP
#############

st.set_page_config(page_title="Drug Repurposing", 
                   page_icon=":pill:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

def p_title(title):
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">{title}</h3>', unsafe_allow_html=True)

#########
#SIDEBAR
########

st.sidebar.header('I want to:')
nav = st.sidebar.radio('',['Go to homepage', 'Get drugs recommendations', 'Visualize graph'])
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

#CONTACT
########
expander = st.sidebar.expander('Contact')
expander.write("I'd love your feedback :smiley: Want to collaborate? Develop a project? Find me on [LinkedIn](https://www.linkedin.com/in/lopezyse/), [Twitter](https://twitter.com/lopezyse) and [Medium](https://lopezyse.medium.com/)")

#######
#PAGES
######

#HOME
#####

if nav == 'Go to homepage':

    st.markdown("<h1 style='text-align: center; color: gray; font-size:28px;'>Drug Repurposing using Knowledge Graph Embeddings</h1>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center; font-size:56px;'<p>&#129302;</p></h3>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center; color: grey; font-size:20px;'>Summarize, paraphrase, analyze text & more. Try our models, browse their source code, and share with the world!</h3>", unsafe_allow_html=True)
    """
    [![Star](https://img.shields.io/github/stars/dlopezyse/Drug-Repurposing-using-KGE.svg?logo=github&style=social)](https://github.com/dlopezyse/Drug-Repurposing-using-KGE)
    [![Follow](https://img.shields.io/twitter/follow/lopezyse?style=social)](https://www.twitter.com/lopezyse)
    """
    st.markdown('___')
    st.write(':point_left: Use the menu at left to select a task (click on > if closed).')
    st.markdown('___')
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>What is this App about?<b></h3>", unsafe_allow_html=True)
    st.write("This is a solution to showcase some of the results produced in the paper [Drug Repurposing Using Knowledge Graph Embeddings with a Focus on Vector-Borne Diseases: A Model Comparison](https://link.springer.com/chapter/10.1007/978-3-031-40942-4_8) as developed by [Diego López Yse](https://www.linkedin.com/in/lopezyse/) and [Diego Torres](https://www.linkedin.com/in/ditorres/) for the Conference on Cloud Computing, Big Data & Emerging Topics 2023.")
    st.write("Drug repurposing methods can identify already approved drugs to treat them efficiently, reducing development costs and time. At the same time, knowledge graph embedding techniques can encode biological information in a single structure that allows users to operate relationships, extract information, learn connections, and make predictions to discover potential new relationships between existing drugs and vector-borne diseases.")
    st.write("In this project, we compare seven knowledge graph embedding models (TransE, TransR, TransH, UM, DistMult, RESCAL, and ERMLP) applied to Drug Repurposing Knowledge Graph (DRKG), analyzing their predictive performance over seven different vector-borne diseases (dengue, chagas, malaria, yellow fever, leishmaniasis, filariasis, and schistosomiasis), measuring their embedding quality and external performance against a ground-truth.")


#-----------------------------------------

#DRUG RECOMMENDATION
####################

if nav == 'Get drugs recommendations':
    st.text('')
    p_title('Get drugs recommendations')
    st.text('')

    disease_selection = st.selectbox("Select disease", ["Dengue", "Chagas", "Malaria", "Yellow Fever", "Leishmaniasis", "Filariasis", "Schistosomiasis"])
    model_selection = st.selectbox("Select embedding model", ["TransE", "TransR", "TransH", "UM", "DistMult", "RESCAL", "ERMLP"])

    if st.button("Get recommendations"):
        #Drug recommendations
        final_selection = disease_selection + model_selection
        ranking_file = pd.read_csv('embedding_models/' + final_selection + '.csv',  sep = ',')

        st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Drugs recommendations<b></h3>", unsafe_allow_html=True)
        st.write(ranking_file)
        st.write('The interpretation of the values in the "score" column is model-dependent, and usually it cannot be directly interpreted as a probability.')
        st.write('The "in_clinical_trials" column states "yes" if the compound exists in ClinicalTrials.gov for the target diseases, and "no" otherwise.')

        #Disease direct connections in DRKG
        st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Direct connections in DRKG for the target disease<b></h3>", unsafe_allow_html=True)
        st.write('Find below all entities in GNBR subgraph that are directly linked to the target disease. Compounds are highlighted in red.')

        import pandas as pd
        import streamlit.components.v1 as components

        #Visualize
        HtmlFile = open('graphs/knowledge_graph_' + disease_selection + '.html', 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height = 625,width=750)

        #Model performance
        performance_file = pd.read_csv('embedding_models/performance_metrics.csv', sep = ';')
        # Filter rows where the 'final_selection' column matches final_selection
        filtered_data = performance_file[performance_file['final_selection'] == final_selection]
        st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Model performance<b></h3>", unsafe_allow_html=True)
        st.dataframe(filtered_data[['Measure', 'Value']], hide_index=True)


#VISUALIZE GRAPH
####################

if nav == 'Visualize graph':
    st.text('')
    p_title('Visualize graph')
    st.text('')
    number_triples = st.selectbox('Select number of triples to sample from GNBR subgraph', ('50', '100', '250', '500'))

    st.write('Zoom in to see graph details')

    import pandas as pd
    import streamlit.components.v1 as components

    #Show GNBR graph

    HtmlFile = open('graphs/knowledge_graph_gnbr_filter_' + number_triples + '.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 625,width=750)
