What does an embedding represent conceptually?
A) embedding is converting raw texts from parsing into a group of numbers called vectors where semantically similar texts are placed closer together for the model to comprehend and proceed further

How does semantic similarity differ from keyword or string matching?
A)keyword similarity just check how many words are matching in the query and the document whereas semantic similarity involves embedding the document and query which represent meaning rather than exact wordsmand compares the embedded values. If the comparasion is high then the similarity score is high and the document is retrieved

Why can two pieces of text with no shared words still be considered similar?
A)Because embeddings capture meaning rather than exact words and coz similarity invloves converting them into vectors and comparing them to get a similarity score which if it is high then the document is retrieved. SO for 2 pieces to be similar the keyword shouldnt be necessarily matching , their similarity scores matching should be enough'

Why are embeddings suitable for retrieval but not for generation?
A)Embeddings compress meaning into vectors.They are excellent for measuring similarity and retrieving relevant documents.However, embeddings do not contain the detailed linguistic structure needed to generate fluent text. Generation requires a language model that predicts words and sentences. So basically embedding is for just embedding and storing in vector db and cannot predict words or text.

Why can’t an LLM reliably “search” large document collections on its own?
A)An LLM can only work with information that is either inside its training data or inside its current context window threfore it cannot efficiently search thousands or millions of external documents by itself.

What role does retrieval play in scaling knowledge beyond a model’s context window?
A)it retrieves only important documents relevant to the query. it doenst place the entire document but splits it up into chunks and sends the important chunks to the llm. This allows access to knowledge much larger than the model's context window

What are the tradeoffs between retrieving too little vs too much information?
A)Too little-can cause important points to be left out , answer becomes incomplete and recall decreases
Too much-can cause irrelevant data to be retrieved, can cause noise in the prompt and can confuse the llm 

How might poor retrieval negatively affect downstream generation?
A)if the retrieval is poor or more than sufficient then it can cause irrelevant,incomplete or incorrect chunks to be generated which can lead to noise, incomplete output and confusion to LLM

What problem does RAG solve that plain prompting cannot?
A)Plain prompting uses model memory and user prompt while RAG uses external knowledge retrieval which is used to answer the prompts outside the models training data

How is RAG different from simply pasting documents into a prompt?
A)Pasting requires user effort and wastes context window and can be time consuming
  RAG retrieves important doucments which are relevant to the query which saves time and space in the contect window and send important info to LLM

Why is it important that the model answers only using retrieved context?
A)Coz there is a chance that it can give the wrong answers in a pre written or memorised format to make it look convincing for the user. Restricting answers to retrieved context reduces hallucinations and ensures responses are grounded in the source documents

Where does hallucination still occur even in RAG systems?
A)when retreival returns irrelevant documents, misses important documents, when LLM misunderstands or when model invents info not present in any document

Which parts of a RAG system are deterministic, and which are probabilistic?
A)document loading, chunking, embedding generation (usually), vector storage ,similarity computation are deterministic as they return the same o/p mostly when given the same i/p
LLM generation is probabilistic as the the next word can change depending on the probability of each word

Where can failures occur in the pipeline even if individual components work well?
A)when the chunking is done incorrectly, embedding doesnt give out proper meaning of the text, when wrong documents are retrieved, LLM misunderstands or when useful context is emitted during prompt construction

Why is separation between retrieval and generation important for debugging?
A)to determine the origin of the error, for example if an error occurs retrieved chunks are checks 
If retrieval is wrong then its a retrieval problem else its a generation problem