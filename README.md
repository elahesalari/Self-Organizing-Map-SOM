# Self-Organizing-Map-(SOM)
<b> Text Clustering using SOM </b>

Text clustering is an unsupervised process, used to separate a document collection into some 
clusters on the basis of the similarity relationship between documents in the collection. Suppose 
𝐷 = {𝑑1, … , 𝑑𝑁} be a collection of 𝑁 documents to be clustered. The task is to divide 𝐷 into 𝑘
clusters 𝐶1, … , 𝐶𝑘 where 𝐶1 ∪ … ∪ 𝐶𝑘 = 𝐷 and 𝐶𝑖 ∩ 𝐶𝑗 = ∅, for 𝑖 ≠ 𝑗.
<br/>
SOM text clustering can be done in two main phases. The first phase is document preprocessing,
which uses Vector Space Model (VSM) to generate a numeric vector for each text document. In 
the next phase, SOM is applied on the document vectors to obtain document clusters. 
<br/>
<br/>
<b> Dataset </b>
<br/>
In this project, we train and test an SOM network to do cluster analysis of a news 
collection, from the BBC news website corresponding to stories in five topical areas from 2004-
2005. This dataset is a collection of 2225 news document, categorized into 5 classes of 
‘business’, ‘entertainment’, ‘politics’, ‘sport’, and ‘tech’.

<b> Phase 1: Document Preprocessing </b>
<br/>
By means of VSM, each document 𝑑𝑖 can be represented by an 𝑛-dimensional feature vector
𝒗𝑖 =< 𝑣𝑖1, … , 𝑣𝑖𝑛 >, where 𝑣𝑖𝑗 is a representation of term 𝑡𝑗
in document 𝑑𝑖 and 𝑛 is the number 
of distinct terms in the document collection 𝐷. 
An approach for computing 𝑣𝑖𝑗 is the Term Frequency - Inverse Document Frequency (TF-IDF) 
weighting scheme. This method computes 𝑣𝑖𝑗 for term 𝑡𝑗
in document 𝑑𝑖 as:
<br/>
![formula](https://user-images.githubusercontent.com/91370511/159134697-02c91891-bf44-47a4-97dc-c3ea5f097b42.PNG)
<br/>

where 𝑡𝑓𝑖𝑗 is the frequency of term 𝑡𝑗 in document 𝑑𝑖, and 𝑑𝑓𝑗 is the number of documents in 𝐷 containing term 𝑡𝑗.
Read ‘bbc-text.csv’ file and for each document: 
  1. Remove all non-letter characters from the documents.
  2. Extract all words of the document and remove the short words (length ≤ 2).
  3. Remove all stop words (e.g., ‘a’, ‘and’, ‘what’, …), given in file ‘stopwords.txt’.
  4. Compute the feature vector for each document, using TF-IDF weighting scheme.

<b> Phase 2: SOM Clustering </b>


a) Winner-takes-all approach
  1. Using all documents, build an SOM with one neuron for each class.
  2. Depict the SOM-hits plot.
  3. Compute and report the confusion matrix.

b) On-center, off-surround approach
  1. Using all documents, build an SOM with 3x3 neurons.
  2. Depict the SOM-hits plot.
  3. Compute the Euclidean distance of all documents to their winner neurons and sum up the 
  distances.
  4. Repeat steps 1-3 for 4x4 and 5x5 topologies.
  5. Report and discuss the overall distances of three topologies.
