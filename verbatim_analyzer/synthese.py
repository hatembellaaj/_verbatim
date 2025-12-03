import pandas as pd

def generer_tableau_synthese_multi_criteres(df_result):
    """
    Crée un tableau de synthèse regroupant les verbatims par Cluster_Label, SubCluster_Label et Satisfaction.
    """
    synthese = df_result.groupby(['Cluster_Label', 'SubCluster_Label', 'Satisfaction']).size().reset_index(name='Nombre')
    tableau = synthese.pivot_table(index=['Cluster_Label', 'SubCluster_Label'], columns='Satisfaction', values='Nombre', fill_value=0)
    tableau['Total'] = tableau.sum(axis=1)
    tableau = tableau.sort_values(by='Total', ascending=False)
    return tableau
