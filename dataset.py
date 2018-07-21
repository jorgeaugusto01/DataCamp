#!/usr/bin/env python
# -*- coding: utf-8 -*-

def printDataSetDetails(dataset):
    #Imprime o tipo do dataset, BUNCH quer dizer que é similar a um dicionário com valor e chave
    print("=================================================")
    print("Dataset description:")
    print(dataset.DESCR)
    print("Dataset type:")
    print(type(dataset))
    print("\nData type:")
    print(type(dataset.data))
    print("\nTargets type:")
    print(type(dataset.target))
    print("\nFeature names:")
    print(dataset.feature_names)
    print("\nDataset keys:")
    print(dataset.keys())
    #Lista a quantidade de linhas e colunas
    print("\nDataset data shape [rows, column]:")
    print(dataset.data.shape)
    print("\nDataset target shape [rows, column]:")
    print(dataset.target.shape)
    #Imprime os tipos de classificações
    print("\nTarget Names: ")
    print(dataset.target_names)
    print("=================================================")
