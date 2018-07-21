# data description
In our task, we need to prepare both Knowledge Base and Recommendation System data.

## KB data
For KB data, we adopt the one-time FREEBASE dump consisting of 63 million.
### Knowledge Graph Acquisition process
1. Link construction. Search Entity with Google Knowledge Graph API(take music/movie/book name as key), we link item in RS(recommender system) to entity in KB. API address(https://developers.google.com/knowledge-graph/).
2. Deduplication. Delete link of item with not single entity or confirm the linkage with addtional accurate information(music's artist/movie's publish year).
3. Expansion. Find triples that contain at least one entity of the seed entity set(linked entity after deduplication), and delete meaningless triples(select relation/restrict entity occurences). Full data dump of Freebase can be download here(https://developers.google.com/freebase/).
4. repeat expansion until you think it's fit for your task(seed entity set becomes larger every time you repeat).

## RS data
For RS data, we use four datasets. Here, we take ml-1m dataset as an example.

