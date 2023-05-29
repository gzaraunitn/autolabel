python3 ../train.py \
  setting=hmdb_ucf \
  attributes.n_attributes=6 \
  attributes.tf_idf_threshold=0.5 \
  attributes.matching_threshold=0.5 \
  attributes.selection=topk attributes.k_clustering=45 \
  attributes.tf_idf_topk_source=5 \
  attributes.tf_idf_topk_target=5 \
  attributes.final_prompt_length=3 \
  loss.target.weight=1.0 \
  loss.target.filtering=top_k_confident_samples \
  loss.target.k=20 \
  loss.target.k_type=percentage \
  data.batch_size=48 \
  solver.type=cosine \
  solver.epochs=20 \
  solver.lr=6.e-6 \
  logging.comet=True \
  logging.tag=hmdb_ucf \
  logging.project_name=autolabel
