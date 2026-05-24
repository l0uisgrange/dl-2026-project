from src.load_config import load_config
from src.train import train 

def main():
    cfg=load_config()
    train(cfg)
    
if __name__ == "__main__":
    main()