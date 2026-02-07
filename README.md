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
