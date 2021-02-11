# -*- coding: utf-8 -*-
"""Executar este arquivo para adicionar ou remover dependências.
Uma dependência é simplesmente a lista de variáveis que devem ser consideradas
na entrada do modelo neural respectivo a uma das saídas do sistema.
É uma etapa opcional, para quando se sabe de antemão quais variáveis
influenciam numa determinada saída. Ler o README para mais informações.
"""
from .data_utils import load_pickle, save_pickle


def add_dependency(output, dependency):
    """Função para informar a dependência de uma variável. 'dependency' é uma
    lista com as variáveis que influenciam na saida informada.

    Carrega o training_dictionary e escreve na entrada "dependency mask" da
    respectiva saída, salvando-o em seguida.

    Args:
        output (str): saída a ser informada a dependência, exemplo: 'y1'
        dependency (list): variável influentes, exemplo: ['u4', 'y1', 'y3']
    """
    # Não se preocupar se o dicionário ainda não existe, neste caso
    # load_pickle retorna um objeto vazio.
    training_dictionary = load_pickle("..\\analysis\\dictionary.pickle")

    try:
        training_dictionary[output]["dependency mask"] = dependency
    except KeyError:
        training_dictionary[output] = {"dependency mask": dependency}

    save_pickle(training_dictionary, "..\\analysis\\dictionary.pickle")

    print("Added dependency = " + str(dependency) + " for output " + output)

    return "Done"


def remove_dependency(output):
    """Remove a dependência da saída informada ou de todas as saídas de uma
    só vez.

    Args:
        output (str): saída cuja dependência deve ser removida
    """
    training_dictionary = load_pickle("..\\analysis\\dictionary.pickle")

    if output == "all":
        for output in training_dictionary:
            try:
                del training_dictionary[output]["dependency mask"]
            except KeyError:
                continue
    else:
        try:
            del training_dictionary[output]["dependency mask"]
        except KeyError:
            return "Could not remove dependency because output " + \
                  f"'{output}' was not found"

    return "Done"

    save_pickle(training_dictionary, "..\\analysis\\dictionary.pickle")


if __name__ == "__main__":
    print("Do you want to add or remove dependencies?")
    option = input()

    if option == "remove":
        print("Inform the output or enter 'all' to remove all dependencies")
        output = input()
        print(remove_dependency(output))

    elif option == "add":
        print("Inform the output to add dependency (example: y1")
        output = input()
        print("Inform the list of influent variables (example: u1, u3, y2)")
        dependency = input().split(', ')
        print(add_dependency(output, dependency))
    else:
        print("Unknown command, please run again and enter 'add' or 'remove'")
