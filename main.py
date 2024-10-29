import subprocess

def main():
    while True:
        print("Select where to run script:")
        print("1. Web")
        print("2. Software")
        
        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            subprocess.run(['python', 'web.py'])
        elif choice == '2':
            subprocess.run(['python', 'local.py'])
        else:
            print("Invalid choice. Please select 1 or 2.")
if __name__ == "__main__":
    main()
