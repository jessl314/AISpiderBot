# AISpiderBot


VIRTUAL ENVIRONMENT

python -m venv venv

venv\Scripts\activate
Powershell: .\venv\Scripts\activate

use python -m pip install package-name 
- if import didn't work


mac: source venv/bin/activate


pip install -r requirements.txt


notes from zims code (idk, if this is right):

- need to make sure action is vector of continuous values



Other things to check:
- where is the loss implemented, check this is optimal
- check optimizer.step, loss.backward, check optimizer.zero_grad 
- also need to check training loop code overall, check in with Cady, see if there is anything simple is left that can be assigned
- matplotlib? -> for creating graphs/visualizing how the model is being trained. like visually seeing the loss function etc. great for testing! -> like in CS 178 :)

- libraries to check: lidar, laspy

How to implement your own code:
if you guys want to mess with the code without messing things up (also assuming you have vscode and git) do the following:
- make your own branch (git checkout <your_name>  -> replace "<your_name>" with your name). run the same command to make sure you are on it. can also make in github.
- if you already have a branch run -> git pull origin main  -> this gets the most recent changes that I just merged onto main branch
- you are now on your own branch and allowed to do literally anything without affecting the code that we'll end up using
- if you all want to make changes, github desktop may be good to download as it is an interface where you can "preview pull request" a.k.a pre-check if your code will merge without conflict. also easier than constantly searching the commands lol.
