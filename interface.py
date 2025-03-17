from main import load_emails, train_model, analyze_emails

def interact():
    have_model = input("Do you have a pretrained model? (y/n): ")
    if have_model.lower() == "y":
        model_dir = input("Model location: ")
        model = None
    else:
        dir = input("Directory of email files (.pst): ")
        outputdir = input("Output model location: ")
        outputstats = input("Output statistics location: ")
        gpu = input("Use GPU? (y/n): ")

        print("Analyzing your data ...")
        data = load_emails(dir)
        stats = analyze_emails(data, outputstats)

        print("Training the model ...")
        model = train_model(data, gpu=(gpu.lower()=="y"), outputdir=outputdir)
    agent(model)
    
def agent(model):
    pass

if __name__ == "main":
    interact()