# How to Setup DRAC

## Initial Setup
### 1. Get account details
Go to https://ccdb.alliancecan.ca/ and setup an account. Once approved, you will receive an email.
Save the username and password used to create the account. You will need this to login into the server.

### 2. SSH Into Server
Open a command window and run the following command:
```
ssh -Y <username>@beluga.computecanada.ca
```

The first time you login, you will be prompted to setup MFA.
Follow the instructions and connect your phone to setup MFA.
If done properly, you should see the following screen:
![success_image](./images/success_drac_login.jpg)

### 3. Clone the repository
Close the repository using the following command:
```
git clone https://github.com/LucasHartmanWestern/rl-for-vrp-csp.git
```


### 4. Install the dependencies
Run the following command to install the dependencies:
```
pip install -r requirements.txt
```

## (Optional) Setting up SSH Keys for easier access
Ref: https://docs.alliancecan.ca/wiki/SSH_Keys

### 1. Generate an SSH key
Assuming ssh-keygen is already installed, use the following command to generate an SSH key:
```
ssh-keygen -t rsa -b 4096 -C "<your_email>"
```

### 2. Add the SSH key to your ccdb account
Go to https://ccdb.computecanada.ca/ssh_authorized_keys and login with your CCDB credentials.
Paste the contents of the SSH key into the text box and click "Add Key".

### 3. Configure private key locally
In your .ssh folder, create a new file called config and add the following:
```
Host beluga
    HostName beluga.computecanada.ca
    User <your_username>
    IdentityFile ~/.ssh/<your_private_key>
```

### 4. Test the SSH connection
Run the following command to test the SSH connection:
```
ssh beluga
```


## Running Experiments

TBD...
