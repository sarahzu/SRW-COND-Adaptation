from GraphFactory import GraphFactory

labeled_graph = GraphFactory('labeled_graph_trump')

csv_path = './data_2/clean_data/election_debate_leaning_2016.csv'
graph_path = './data_2/clean_data/election_debate_leaning_2016.graphml'
ext_graph_path = './data_2/clean_data/extended_graph.graphml'


labeled_graph.create_graphs_from_dataframe('QT_REP',csv_path)
labeled_graph.prune_graph(1,2,2,True)
labeled_graph.print_info()


labeled_graph.extend_labeled_graph()

feature_vector = labeled_graph.create_feature_vector()
