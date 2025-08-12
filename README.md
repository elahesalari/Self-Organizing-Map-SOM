# 🧠 Self-Organizing Map (SOM)

## 📄 Text Clustering using SOM

Text clustering is an unsupervised learning technique that groups a collection of documents into clusters based on their similarity. Given a document collection \( D = \{d_1, \ldots, d_N\} \), the goal is to partition \( D \) into \( k \) clusters \( C_1, \ldots, C_k \) such that \( C_1 \cup \ldots \cup C_k = D \) and \( C_i \cap C_j = \emptyset \) for \( i \neq j \).

SOM-based text clustering proceeds in two phases:  
- **Phase 1:** Document preprocessing using the Vector Space Model (VSM) to convert text documents into numeric vectors  
- **Phase 2:** Applying SOM on these vectors to generate document clusters

---

## 🗃️ Dataset

We used the BBC news dataset containing 2,225 news documents categorized into 5 classes:  
`business`, `entertainment`, `politics`, `sport`, and `tech`. The dataset covers stories from 2004-2005.

---

## 🔍 Phase 1: Document Preprocessing

Each document \( d_i \) is represented as an \( n \)-dimensional feature vector  
\[
\mathbf{v}_i = \langle v_{i1}, \ldots, v_{in} \rangle
\]  
where \( v_{ij} \) represents the weight of term \( t_j \) in document \( d_i \), and \( n \) is the total number of distinct terms in the collection.

We computed \( v_{ij} \) using the Term Frequency-Inverse Document Frequency (TF-IDF) weighting scheme:

<p align="center">
  <img src="https://user-images.githubusercontent.com/91370511/159134697-02c91891-bf44-47a4-97dc-c3ea5f097b42.PNG" alt="TF-IDF formula" />
</p>

where:  
- \( tf_{ij} \) is the frequency of term \( t_j \) in document \( d_i \),  
- \( df_j \) is the number of documents in which term \( t_j \) appears

Preprocessing steps included:  
1. Removing all non-letter characters  
2. Extracting words and removing short words (length ≤ 2)  
3. Removing stop words (from the provided `stopwords.txt`)  
4. Computing TF-IDF vectors for each document

---

## 🧩 Phase 2: SOM Clustering

### (a) Winner-takes-all Approach  
- Built an SOM with one neuron per class using all documents
- Visualized SOM hits plot  
- Computed the confusion matrix for classification accuracy

### (b) On-center, Off-surround Approach  
- Built SOMs with 3×3 neurons using all documents  
- Visualized SOM hits plot  
- Computed the sum of Euclidean distances between documents and their winning neurons  
- Repeated the process for 4×4 and 5×5 SOM topologies  
- Compared the overall distances to evaluate cluster compactness across topologies

