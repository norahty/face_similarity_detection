# face_similarity_detection

<img width="866" alt="Screen Shot 2023-09-13 at 8 45 11 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/8f631f3d-b2e3-4a0f-99ba-6694badbd0d0">


This project uses t-SNE to plot faces of paintings using face recognition software
We are expecting the t-SNE plot our algorithm is creating to show different clusters of facial data analysis for artists with different styles with the following evaluation matrices:
The more scattered the clusters are, the smaller the similarity correlation between the compared clusters is. The more united one specific cluster is, the higher the similarity correlation within this one cluster is.

<img width="924" alt="Screen Shot 2023-09-13 at 8 48 47 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/122585bf-fdf4-4455-9b1c-bedac57bbfce">

# To run this project on google colab: [https://drive.google.com/drive/folders/1-yMU9guWl4pLbCeqsbxAZKSxJJXIm5Gq?usp=drive_link](https://drive.google.com/drive/folders/1-yMU9guWl4pLbCeqsbxAZKSxJJXIm5Gq?usp=share_link)

# Data Training
In the dataset training phase, we deliberately selected a dataset consisting of 21-year-old individuals, encompassing 188 faces representing White, Asian, and Black demographics. Consequently:

A higher degree of similarity correlation between an artist's dataset and the natural faces dataset suggests that the artist's style leans towards a more photorealistic portrayal.

Greater clustering within a specific cluster indicates a higher level of racial homogeneity among the characters created by that artist.
![image](https://github.com/norahty/face_similarity_detection/assets/94091909/aaf28289-20b8-478d-8bfc-a1be8fc6882e)

# Implications of the Experiment
Botero's artistic style stands out as highly distinctive, setting it apart from the majority of art styles found throughout history. This distinction is particularly evident since Botero's characters consistently adhere to the artistic principles of primitivism.

In contrast to Botero and Kahlo, the majority of the other artists in the dataset created works predominantly featuring white characters, reflecting a racial focus prevalent during their respective historical periods. Additionally, artists from an older dataset primarily depicted white characters, resulting in a dataset characterized by racial homogeneity and consequently leading to denser plot clusters.

This observation holds unless an artist intentionally deviates from creating highly realistic portraits or opts for a more diverse or unconventional choice of subject matter compared to prevailing artistic norms of the past.

One particularly interesting observation is that most of the paintings we examined did not coincide with a small cluster of natural faces. This intriguing phenomenon could potentially offer valuable insights into the distinct artistic styles employed by the selected artists.

<img width="277" alt="Screen Shot 2023-09-13 at 8 46 34 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/9b6dca04-fa2f-484c-b23f-e1ef3964bc43">
<img width="286" alt="Screen Shot 2023-09-13 at 8 49 21 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/233193fe-509a-466f-a372-4fa7b46f0b6a">

<img width="915" alt="Screen Shot 2023-09-13 at 8 53 45 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/6cb7bcd6-0b11-4dc2-a990-0bc621c5dca8">
<img width="915" alt="Screen Shot 2023-09-13 at 8 53 57 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/7d7443bc-06d1-465b-895d-02d8c0b89633">
<img width="915" alt="Screen Shot 2023-09-13 at 8 54 10 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/11c64524-1948-425d-8421-2e2886c341c4">
<img width="915" alt="Screen Shot 2023-09-13 at 8 54 39 AM" src="https://github.com/norahty/face_similarity_detection/assets/94091909/1c8acea8-3480-4430-a94a-c7d312478407">
Run all the code in the copied file and change perplexity (grid [20]), artists(grid [7]) as you like to see different results

# What to do next:
We want to try distinguishing different faces of the paintings using machine learning methods
We are planning on starting by dividing each dataset at a 8:2 ratio as for training and testing in our implementation for the second stream of our implementation.
We will select facial features from the dataset, determine importance of each feature, learning similarities between portrait pairs and compare the result after training


