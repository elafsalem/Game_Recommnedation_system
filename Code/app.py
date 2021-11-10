import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel, paired_cosine_distances, cosine_similarity
from nltk.corpus import stopwords
import nltk
import streamlit as st
import shap
import streamlit.components.v1 as stc
import pickle
import time
import _pickle as cPickle
# import cPickle as pickle
import gc


# In[4]:


merge_dataset = pd.read_csv("/Users/muntaha/Documents/Project4_v2/Game_Recommnedation_system/Data/merged_description.csv", index_col= 0)
game_images = pd.read_csv("/Users/muntaha/Documents/Project4_v2/new_game_data/steam_media_data.csv")


# In[5]:


# merge_dataset.head(1)


# In[6]:


new_data = game_images.rename(index=str, columns={"steam_appid":"appid"})
#merge two datasets
merge_dataset_image = merge_dataset.merge(new_data, on="appid")


# In[16]:


# for x in merge_dataset_image.head(1).header_image:
#     print(x)


# In[17]:


# header_image
# screenshots
# background

for x in game_images.head(2).header_image:
    print(x)
    


# ## Load TFIDF

# In[9]:


# tfidf_matrix = pickle.load(open("/Users/muntaha/Documents/Project4_v2/Game_Recommnedation_system/Code/tfidf_matrix.pkl", 'rb'))


# In[10]:


# tfidf_matrix.shape


# # Model

# In[38]:


# def content_recommender(name, games, similarities, vote_threshold=1000, rating_threshold=0.7) :
    
#     # Get the game by the title
#     game_index = games[games['name']==name].index
    
#     # Create a dataframe with the game id, name, and rating information with similarity
#     sim_df = pd.DataFrame(
#         {'appid': games['appid'],
#          'game': games['name'], 
#          'similarity': np.array(similarities[game_index, :].todense()).squeeze(),
#          'diversity': 1- np.array(similarities[game_index, :].todense()).squeeze(),
#          'vote_count': games['total_ratings'],
#          'percent_positive_ratings': games['percent_positive_ratings']
#         })
    
#     # Get the top 10 games that satisfy our thresholds
#     top_games = sim_df[(sim_df['vote_count']>vote_threshold) & 
#                        (sim_df['percent_positive_ratings']>rating_threshold)].sort_values(by='similarity', ascending=False).head(10)
    
#     return top_games


# In[12]:


# def content_recommender2(name_list,list_game_type ,games, similarities, vote_threshold=1000, rating_threshold=0.7) :
#     i = 0
#     j = 0
    
# #     converting list of type into string
#     string_type = ""
#     for x in list_game_type:
#         string_type += " "+ x
        
#     recomended_game_name = {}
#     game_index = []
    
#     top_games = pd.DataFrame()
    
# #     find name index that is same game type 
#     for name in range(len(games)):
#         st.write(name)
#         if list(games.iloc[games[games['name']==name_list[i]].index].genres)[0] in list_game_type and i < len(name_list:
#             game_index.insert(j,games[games['name']==name_list[i]].index)
#             i = i+1
#             j = j+1
#             st.write("inside name method")
        
#         else:
#             i = i+1
#             st.write("inside continue")
#             continue
    
#     for x , y in zip(range(len(name_list)), range(len(game_index))):
#         sim_df = pd.DataFrame(
#             {'appid': games['appid'],
#              'game': games['name'], 
#              'game_type': games['genres'], 
#              'similarity': np.array(similarities[game_index[y], :].todense()).squeeze(),
#              'diversity': 1- np.array(similarities[game_index[y], :].todense()).squeeze(),
#              'vote_count': games['total_ratings'],
#          'percent_positive_ratings': games['percent_positive_ratings']})
        
        
#         top_games = sim_df[(sim_df.iloc[x].game_type in list_game_type) & (sim_df['vote_count']>vote_threshold) & 
#                            (sim_df['percent_positive_ratings']>rating_threshold)].sort_values(by='similarity', ascending=False).head(10).sort_values(by='percent_positive_ratings',ascending=False)
#         for t,j in zip(top_games.game, top_games.game_type):
#             recomended_game_name[t] = j
        
#         st.write(top_games)
        
#     return recomended_game_name, top_games
def content_recommender2(name_list,list_game_type ,games, similarities, vote_threshold=1000, rating_threshold=0.7) :
    i = 0
    j = 0
    

        
    recomended_game_name = {}
    game_index = []
    
    top_games = pd.DataFrame()
    
#     find name index that is same game type 
    for name in name_list: #2
        if list(games.iloc[games[games['name']==name_list[i]].index].genres)[0]:
            print(list(games.iloc[games[games['name']==name_list[i]].index].genres)[0])
            game_index.insert(j,games[games['name']==name_list[i]].index)
            i = i+1
            j = j+1
        
        else:
            i = i+1
            continue
    
    for x , y in zip(range(len(name_list)), range(len(game_index))):
        sim_df = pd.DataFrame(
            {'appid': games['appid'],
             'game': games['name'], 
             'game_type': games['genres'], 
             'similarity': np.array(similarities[game_index[y], :].todense()).squeeze(),
             'diversity': 1- np.array(similarities[game_index[y], :].todense()).squeeze(),
             'vote_count': games['total_ratings'],
         'percent_positive_ratings': games['percent_positive_ratings']})
        
        
        top_games = sim_df[(sim_df['vote_count']>vote_threshold) &
                           (sim_df['percent_positive_ratings']>rating_threshold)].sort_values(by='similarity', ascending=False).head(8).iloc[1: , :]
        # st.write(sim_df.similarity.iloc[y])



        for t,j in zip(top_games.game, top_games.game_type):
            recomended_game_name[t] = j


        
        
    return recomended_game_name, top_games.head(16)




# In[13]:

top_games = merge_dataset_image[(merge_dataset_image['total_ratings']>1000) & 
                           (merge_dataset_image['percent_positive_ratings']>0.80)].sort_values(by='total_ratings',ascending=False).head(8)

list_of_top_games = list(top_games.name)


def type_based(list_game_type,games,vote_threshold=1000, rating_threshold=0.7):
    index = merge_dataset_image[(merge_dataset_image.genres == list_game_type[0]) & 
                                (merge_dataset_image.total_ratings > vote_threshold) &
                                (merge_dataset_image.percent_positive_ratings > rating_threshold)].sort_values(by='percent_positive_ratings', ascending=False).index
    index = index[0:10]
    dicitt = {}
    for x in index:
        dicitt[merge_dataset_image.name.loc[x]] = merge_dataset_image.header_image.loc[x]
    return dicitt




# # find index of each name to bring the similar game by entring game name
# indexs = pd.Series(merge_dataset.index, index = merge_dataset.name).drop_duplicates()


# In[14]:


# # test recommendtion system based on sigmoid_kernel
# def recomend_game(name, num ,sig=sig):
#     indx = indexs[name]
#     sig_score = list(enumerate(sig[indx])) # get similarity score
#     sig_score = sorted(sig_score, key= lambda x:x[1], reverse = True) # sort game
#     sig_score = sig_score[1:num+1]
    
#     game_index = [i[0] for i in sig_score] # get index of each score
    
#     return merge_dataset.name.iloc[game_index] # return the name


# ### 1) sigmoid_kernel

# In[15]:


# # to find  content based similarity  will try to use sigmoid kernel to find the propabilty of detailed_description similarity
# sig = sigmoid_kernel(tfidf_matrix,tfidf_matrix)
# # to much time but works 


# In[16]:


# recomend_game("PAYDAY 2",3)


# ### 2) cosine_similarity

# In[37]:



# def read_cosine_similarity_model():
#     f = open("/Users/muntaha/Documents/Project4_v2/Game_Recommnedation_system/Code/cosine_similarity_model.pkl", 'rb')
#     gc.disable()
#     similarities = cPickle.load(f)
#     gc.enable()
#     f.close()
#     return similarities


# In[18]:


# similarities = pickle.load(open("/Users/muntaha/Documents/Project4_v2/Game_Recommnedation_system/Code/cosine_similarity_model.pkl", 'rb'))


# In[19]:


# similarities
# list_game_name = ["PAYDAY 2","Counter-Strike"]
# list_game_type = ["Action","Casual","Strategy","Action;RPG"]


# In[20]:


# similar_games = content_recommender("PAYDAY 2", merge_dataset, similarities, 
#                                     rating_threshold=0.80)
# similar_games.sort_values(by='percent_positive_ratings',ascending=False)


# In[21]:


# similar_games = content_recommender("Counter-Strike", merge_dataset, similarities, 
#                                     rating_threshold=0.80)
# similar_games.sort_values(by='percent_positive_ratings',ascending=False)


# In[ ]:





# In[22]:


# similar_games_dict, similar_games_df  = content_recommender2(list_game_name, list_game_type,merge_dataset, similarities, 
#                                     rating_threshold=0.80)
# # similar_games.head(5).sort_values(by='percent_positive_ratings',ascending=False)


# In[23]:


# similar_games_df


# In[24]:


# similar_games_dict


# # web streamlite

# In[33]:



# def main():
#     st.title("Game Recommendation App")
#     st.markdown("""---""")
#     st.write("Choose Games you play")
    
#     game_type()
    
# #     checkbox1 = st.checkbox("Counter-Strike")
# #     checkbox2 = st.checkbox("PAYDAY 2")
  
# #     st.write("state1",checkbox1)
# #     st.write("state2",checkbox2)
# #     time.sleep(2)

   
                        
        
            
            

# def game_type():
#     option = st.selectbox('Choose game type you like:',('Action','Casual','Strategy','Action;RPG'))
#     st.write('You selected:', option)
#     time.sleep(2)
#     return option

# def game_name(g1,g2):
#     listt = []
#     if g1:
#         listt.append("Counter-Strike")
#     if g2:
#         listt.append("PAYDAY 2")
#     time.sleep(2)
#     return listt
        
# def predect_game(list_game_name,list_game_type):
#     st.write("All recommended games:")
#     st.markdown("""---""")
# #     similar_games_dict, similar_games_df  = content_recommender2(list_game_name, list_game_type,merge_dataset, read_cosine_similarity_model(), 
# #                                                                  rating_threshold=0.80)
#     st.write("here")
#     time.sleep(2)
    
    
# def collect_values(checkbox1 = True,checkbox2=True):
#     st.markdown("""---""")
#     if checkbox1 == True or checkbox2 == True:
#         time.sleep(2)
#         list_game_name = game_name(checkbox1,checkbox2)
#         time.sleep(2)
#         list_game_type = game_type()
#         time.sleep(2)
#     if st.button("Recommend"):
#         if selected_types is not None:
#             st.markdown("""---""")
#             time.sleep(2)
#             predect_game(list_game_name,list_game_type)
#         else:
#             st.write("You have to choose atleast one game type!")
            
#     else:
#             st.write("You have to choose atleast one game!")
#     return "done"



# if __name__ == '__main__':
#     main()


# In[15]:



def main():
    
    similarities = read_cosine_similarity_model()

    # page_bg_img = "https://images7.alphacoders.com/557/557051.jpg"
    # st.image(page_bg_img)


    new_title = '<p style="font-family:sans-serif; text-align: center; color:Black; font-size: 42px;">Game Recommendation App</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    st.markdown('##')
    st.markdown('##')




    clicked_button_text = ""

    list_game_selected = []





    # left, right = st.columns(2)
    # with left:
    #     if st.button('Button 1'):
    #         clicked_button_text = "Choose Games you play"
           

    # with right:
    #     if st.button('Button 2'):
    #         clicked_button_text = "Choose Games type you play"


    # st.write(clicked_button_text)


    # st.write("Choose games:")
    text1 = '<p style="font-family:sans-serif; background-color:#6495ED; color:white; font-size: 20px; ">  choose game </p>'
    st.markdown(text1, unsafe_allow_html=True)

    st.markdown('##')


    g1, g2, g3 ,g4 = st.columns(4)

    with g1: 
        if st.checkbox('Counter-Strike'):
            list_game_selected.append('Counter-Strike: Global Offensive')

    with g2: 
        if st.checkbox('Dota 2'):
            list_game_selected.append('Dota 2')

    with g3:
        if (st.checkbox("Team Fortress 2")):
            list_game_selected.append('Team Fortress 2')

    with g4:
        if (st.checkbox("Garry's Mod")):
            list_game_selected.append("Garry's Mod")

    load_images(list_of_top_games[0:4])


    g5, g6, g7 ,g8 = st.columns(4)

    with g5: 
        if st.checkbox('PAYDAY 2'):
            list_game_selected.append('PAYDAY 2')

    with g6: 
        if st.checkbox('Unturned'):
            list_game_selected.append('Unturned')

    with g7:
        if (st.checkbox("Rainbow Six")):
            list_game_selected.append("Tom Clancy's Rainbow Six® Siege")

    with g8:
        if (st.checkbox("Rust")):
            list_game_selected.append('Rust')
    load_images(list_of_top_games[4:])



        
    
    list_game_type = collect_values()  
    if st.button("Recommend"):
        predect_game(list_game_selected,list_game_type,similarities)
        sort_game_type_based(list_game_type)
  
            
            
def load_images(name_list):
    idx = 0 
    i = 0
    filteredImages = []

    for namee in name_list:
        filteredImages.append(merge_dataset_image[merge_dataset_image.name == namee].header_image.iloc[0])


    while idx < len(filteredImages):
        for _ in range(len(filteredImages)):
            cols = st.columns(4) 
            if idx < len(filteredImages):
                cols[0].image(filteredImages[idx], width=150, use_column_width=True)
                idx+=1
                cols[1].image(filteredImages[idx], width=150, use_column_width=True)
                idx+=1
                cols[2].image(filteredImages[idx], width=150, use_column_width=True)
                idx+=1
                cols[3].image(filteredImages[idx], width=150, use_column_width=True)
                idx+=1
       
                

def game_type():
    listt = []
    text2 = '<p style="font-family:sans-serif; background-color:#6495ED; color:white; font-size: 20px; ">  choose game type you like </p>'
    st.markdown(text2, unsafe_allow_html=True)
    option = st.selectbox('',('Action','Adventure','Casual','Indie','RPG','Racing','Simulation',
        'Sports','Strategy'))
    listt.append(option)
    listt.append(option)
    return listt



def game_name(g1,g2):
    listt = []
    if g1:
        listt.append("Counter-Strike")
    if g2:
        listt.append("PAYDAY 2")
    return listt

    

def predect_game(list_game_selected,list_game_type,similarities):
    

    text3 = '<p style="font-family:sans-serif; background-color:#6495ED; color:white; font-size: 20px; ">  All recommended games </p>'
    st.markdown(text3, unsafe_allow_html=True)

    similar_games_dict, similar_games_df  = content_recommender2(list_game_selected, list_game_type,merge_dataset_image, similarities, 
                                                                 rating_threshold=0.80)
    # st.write(similar_games_dict)
    load_prediction_data(similar_games_dict)


def load_prediction_data(similar_games_dict):

    similar_games_list = list(similar_games_dict.keys())

    idx = 0 
    i = 0
    filteredImages = []

    for namee in similar_games_list:
        filteredImages.append(merge_dataset_image[merge_dataset_image.name == namee].header_image.iloc[0])


    while idx < len(filteredImages):
        for _ in range(len(filteredImages)):
            cols = st.columns(4) 
            if idx < len(filteredImages):
                cols[0].image(filteredImages[idx], width=150, use_column_width=True,caption=similar_games_list[idx])
                idx+=1
                if idx < len(filteredImages):
                    cols[1].image(filteredImages[idx], width=150, use_column_width=True,caption=similar_games_list[idx])
                idx+=1
                if idx < len(filteredImages):
                    cols[2].image(filteredImages[idx], width=150, use_column_width=True,caption=similar_games_list[idx])
                idx+=1
                if idx < len(filteredImages):
                    cols[3].image(filteredImages[idx], width=150, use_column_width=True,caption=similar_games_list[idx])
                idx+=1


        




      

def collect_values():
    list_game_type = game_type()  
        
#         if st.button("Recommend"):
#             time.sleep(2)
#             st.markdown("""---""")
#             predect_game(list_game_name,list_game_type)
                 
    return list_game_type

def sort_game_type_based(list_game_type):
    string_type = ""+list_game_type[0] + " Games"

    text4 = '<p style="font-family:sans-serif; background-color:#6495ED; color:white; font-size: 20px; "> '+string_type+' </p>'
    st.markdown(text4, unsafe_allow_html=True)

    dicct = type_based(list_game_type,merge_dataset_image,rating_threshold=0.80)
    load_prediction_data(dicct)


@st.cache(allow_output_mutation = True)
def read_cosine_similarity_model():
    f = open("/Users/muntaha/Documents/Project4_v2/Game_Recommnedation_system/Code/cosine_similarity_model.pkl", 'rb')
    gc.disable()
    similarities = cPickle.load(f)
    gc.enable()
    f.close()
    return similarities



if __name__ == '__main__':
    main()

