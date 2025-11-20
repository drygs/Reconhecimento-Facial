import library


def mostrar_menu():
    print("\n==============================")
    print("       SISTEMA DE ROSTOS  ")
    print("================================")
    print("1 - Tirar fotos")
    print("2 - Gerar/Regerar db")
    print("3 - Iniciar reconhecimento")
    print("0 - Sair")
    print("================================")

def main():
    while True:
        library.os.system("cls||clear")
        mostrar_menu()
        opcao = input("Escolhe uma opção: ").strip()

        if opcao == "1":
            library.captura_fotos.capturar_fotos()

        elif opcao == "2":
            library.reconhecimento.build_db_embeddings()

        elif opcao == "3":
            library.reconhecimento.start_reconhecimento()

        elif opcao == "0":
            print("A sair...")
            break

        else:
            print("Opção inválida, mano. Tenta outra vez.\n")


if __name__ == "__main__":
    main()
