import os

def create_env():
    print("\n🎬 Subtitle Generator Setup")
    print("===========================\n")

    key = input("Enter DEEPSEEK_API_KEY (Optional for better translation): ").strip()
    
    with open(".env", "w") as f:
        f.write(f'DEEPSEEK_API_KEY={key}\n')
        
    print("\n✅ .env file created successfully!")

if __name__ == "__main__":
    create_env()
