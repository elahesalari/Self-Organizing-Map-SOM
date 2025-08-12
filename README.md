# 🧠 Self-Organizing Map (SOM)

**Text Clustering using SOM**

Text clustering is an unsupervised process used to separate a document collection into clusters based on the similarity relationships between documents. Let 𝐷 = {𝑑1, … , 𝑑𝑁} represent a collection of 𝑁 documents to be clustered. The task was to divide 𝐷 into 𝑘 clusters 𝐶1, … , 𝐶𝑘 such that 𝐶1 ∪ … ∪ 𝐶𝑘 = 𝐷 and 𝐶𝑖 ∩ 𝐶𝑗 = ∅ for 𝑖 ≠ 𝑗.

SOM-based text clustering was performed in two main phases:  
- The first phase involved document preprocessing, using the Vector Space Model (VSM) to generate numeric vectors for each text document.  
- In the second phase, SOM was applied on these document vectors to obtain the clusters.

---

## 🗃️ Dataset

The project utilized the BBC news dataset, which contains 2,225 news documents categorized into five classes:  
**business**, **entertainment**, **politics**, **sport**, and **tech**. The dataset covers stories from 2004 to 2005.

---

## 🔍 Phase 1: Document Preprocessing

Using VSM, each document 𝑑𝑖 was represented as an 𝑛-dimensional feature vector  
𝒗𝑖 = < 𝑣𝑖1, … , 𝑣𝑖𝑛 >,  
where 𝑣𝑖𝑗 indicates the weight of term 𝑡𝑗 in document 𝑑𝑖, and 𝑛 is the total number of distinct terms in the document collection 𝐷.

The weights 𝑣𝑖𝑗 were computed using the Term Frequency - Inverse Document Frequency (TF-IDF) weighting scheme as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/91370511/159134697-02c91891-bf44-47a4-97dc-c3ea5f097b42.PNG" alt="TF-IDF formula" />
</p>

where:  
- 𝑡𝑓𝑖𝑗 is the frequency of term 𝑡𝑗 in document 𝑑𝑖  
- 𝑑𝑓𝑗 is the number of documents in 𝐷 containing term 𝑡𝑗  

Preprocessing steps applied to each document included:  
1. Removal of all non-letter characters  
2. Extraction of words and removal of short words with length less than or equal to 2  
3. Removal of stop words, as specified in the provided `stopwords.txt` file  
4. Computation of feature vectors using the TF-IDF weighting scheme

---

## 🧩 Phase 2: SOM Clustering

### a) Winner-takes-all Approach  
- An SOM was built using all documents with one neuron assigned per class  
- The SOM hits plot was depicted  
- The confusion matrix was computed and reported to evaluate classification performance

### b) On-center, Off-surround Approach  
- SOMs were constructed with 3×3 neurons using the entire dataset  
- The SOM hits plot was visualized  
- The Euclidean distances between all documents and their respective winner neurons were calculated and summed  
- The process was repeated for SOM topologies of sizes 4×4 and 5×5  
