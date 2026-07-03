"""
=========================================================
Facial Recognition System
Module: main.py

Description:
Main entry point of the application.

This file provides a user-friendly menu to interact with
the system:

1. Register new user (capture faces)
2. Train model
3. Start recognition
4. Exit
=========================================================
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================

from capture import capture_faces
from train import train_model
from recognition import start_recognition
from utils import print_title

import os

# =============================================================================
# CLEAR SCREEN FUNCTION
# =============================================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# =============================================================================
# MAIN MENU
# =============================================================================

def show_menu():

    clear_screen()

    print_title("FACIAL RECOGNITION SYSTEM")

    print("1. Register new user")
    print("2. Train model")
    print("3. Start recognition")
    print("4. Exit")

    print("\n========================================")

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():

    while True:

        show_menu()

        option = input("Select an option: ").strip()

        if option == "1":

            clear_screen()
            capture_faces()
            input("\nPress ENTER to continue...")

        elif option == "2":

            clear_screen()
            train_model()
            input("\nPress ENTER to continue...")

        elif option == "3":

            clear_screen()
            start_recognition()
            input("\nPress ENTER to continue...")

        elif option == "4":

            print("\nExiting system...\n")
            break

        else:

            print("\n[ERROR] Invalid option.")
            input("Press ENTER to continue...")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
