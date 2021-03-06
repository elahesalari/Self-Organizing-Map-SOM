# Self-Organizing-Map-(SOM)
<b> Text Clustering using SOM </b>

Text clustering is an unsupervised process, used to separate a document collection into some 
clusters on the basis of the similarity relationship between documents in the collection. Suppose 
π· = {π1, β¦ , ππ} be a collection of π documents to be clustered. The task is to divide π· into π
clusters πΆ1, β¦ , πΆπ where πΆ1 βͺ β¦ βͺ πΆπ = π· and πΆπ β© πΆπ = β, for π β  π.
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
βbusinessβ, βentertainmentβ, βpoliticsβ, βsportβ, and βtechβ.

<b> Phase 1: Document Preprocessing </b>
<br/>
By means of VSM, each document ππ can be represented by an π-dimensional feature vector
ππ =< π£π1, β¦ , π£ππ >, where π£ππ is a representation of term π‘π
in document ππ and π is the number 
of distinct terms in the document collection π·. 
An approach for computing π£ππ is the Term Frequency - Inverse Document Frequency (TF-IDF) 
weighting scheme. This method computes π£ππ for term π‘π
in document ππ as:
<br/>
![formula](https://user-images.githubusercontent.com/91370511/159134697-02c91891-bf44-47a4-97dc-c3ea5f097b42.PNG)
<br/>

where π‘πππ is the frequency of term π‘π in document ππ, and πππ is the number of documents in π· containing term π‘π.
Read βbbc-text.csvβ file and for each document: 
  1. Remove all non-letter characters from the documents.
  2. Extract all words of the document and remove the short words (length β€ 2).
  3. Remove all stop words (e.g., βaβ, βandβ, βwhatβ, β¦), given in file βstopwords.txtβ.
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
