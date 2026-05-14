from capture import capture_faces
from train import train_model
from recognition import start_recognition

def show_menu():
    print("\n==============================")
    print(" Facial Recognition System")
    print("==============================")
    print("1 - Register new face")
    print("2 - Train model")
    print("3 - Start recognition")
    print("4 - Exit")
    print("==============================")

while True:
    show_menu()

    option = input("Select an option: ")

    if option == "1":
        capture_faces()

    elif option == "2":
        train_model()

    elif option == "3":
        start_recognition()

    elif option == "4":
        print("Exiting system...")
        break

    else:
        print("Invalid option")
