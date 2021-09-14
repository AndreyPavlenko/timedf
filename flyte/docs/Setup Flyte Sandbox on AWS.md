
## AWS

###    Windows (with GitBash as a default terminal)

## **Installation**

### **Steps 1-5 should be executed in PowerShell as admin**

1.	**Install chocolatey**. Chocolatey is a package manager for Windows. It makes installing other dependencies much simpler. **(do once)**

    >Set-ExecutionPolicy AllSigned

    >Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

2.	**Install Docker Desktop**. To install Docker Desktop on windows follow the instructions on https://docs.docker.com/docker-for-windows/install/. **(do once)**

3.	**Install Terraform**. Terraform is an “infrastructure as code” software tool. It is used to manage clusters on the cloud. **(do once)**
    
    >choco install terraform

4.	**Install AWS CLI**. AWS CLI is used in conjuncture with terraform to manage AWS clusters. **(do once)**
    
    >choco install awscli

5.	**Install kubectl**. Kubectl is a command-line tool for Kubernetes. It is used to manage EKS clusters. **(do once)**
    
    >choco install kubernetes-cli

    ### **Starting from step 6 all steps should be executed in Git Bash**

6.	**Create Python Virtual Environment**. Python virtual environment are used to keep dependencies of multiple projects separate. **(do once)**

    >pip install virtualenv

    >virtualenv venv

7.	**Activate Python Virtual Environment**.
    
    >source ~/venv/Scripts/activate

8. **Write KUBECONFIG and FLYTECTL_CONFIG in .bashrc** You need to manually set environment variables KUBECONFIG and FLYTECTL_CONFIG (On windows they are automatically set up incorrectly). **(do once)**
   
    >echo -e "export KUBECONFIG=$HOME/.kube/config:$HOME/.flyte/k3s/k3s.yaml\nexport FLYTECTL_CONFIG=$HOME/.flyte/config-sandbox.yaml" >> ~/.bashrc
    
9.	**Install flytekit and scandir**. Flytekit is a Python library for interacting with Flyte tasks, workfows and launch plans. **(do once)**

    >pip3 install flytekit

    >pip3 install scandir

10.	**Install flytectl**. Flytectl is a command-line interface for Flyte. **(do once)**

    >curl -s https://raw.githubusercontent.com/lyft/flytectl/master/install.sh | bash

    >echo -e "export PATH=$(pwd)/bin:$PATH" >> ~/.bashrc

11. **Create Flyte sandbox config**. Create file *$HOME/.flyte/config-sandbox.yaml* with this content(**do once**):

<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>
admin:
  # For GRPC endpoints you might want to use dns:///flyte.myexample.com
  endpoint: localhost:30081
  authType: Pkce
  insecure: true
logger:
  show-source: true
  level: 0
storage:
  connection:
    access-key: minio
    auth-type: accesskey
    disable-ssl: true
    endpoint: http://localhost:30084
    region: us-east-1
    secret-key: miniostorage
  type: minio
  container: "my-s3-bucket"
  enable-multicontainer: true

</pre></div>

\*This file is created automatically when sandbox is created locally with command  *flytectl sandbox start*

12.	**Configure AWS CLI**. Use existing or create a new access key using AWS Management Console. Optionally, set default region and output format.**(do once)**

    >aws configure

    Example:
    >AWS Access Key ID: \<AWS Access Key ID>

    >AWS Secret Access Key: \<Secret Access Key>

    >Default region name: [us-east-2] 

    >Default output format: [json]

## **Usage**

1.	**Create docker image and upload it to a registry**. You can add your own workflows to myflyteapp/myapp/workflows folder. The registry where you upload your image must be accessible by AWS EKS cluster(for example: DockerHub).

    >git clone https://github.com/flyteorg/flytekit-python-template.git myflyteapp

    >cd myflyteapp
    
    Add your workflow to *myflyteapp/myapp/workflows*. Add required packages to *myflyteapp/requirements.txt*

    >docker build . --tag \<registry/repo:version>

    >docker push \<registry/repo:version>

    The version in _flytectl register files_ does not need to match the version of the image, but it is recommended.

2.	**Create Kubernetes cluster**. Download a terraform configuration for creating AWS EKS cluster. You can change the number of instances or their types by changing values *asg_desired_capacity* and *instance_type* in *learn-terraform-provision-eks-cluster/eks-cluster.tf*

    >git clone https://github.com/hashicorp/learn-terraform-provision-eks-cluster.git

    >cd learn-terraform-provision-eks-cluster

    Edit /eks-cluster.tf 

    >terraform init -upgrade

    >terraform apply -auto-approve

3. **Add the created cluster to kubeconfig**

    >aws eks --region \$(terraform output -raw region) update-kubeconfig --name \$(terraform output -raw cluster_name)
    
    The command above will output context that you need to swich to using: 
    
    >kubectl config set-context \<your-context>

4.	**Install Flyte Sandbox on the cluster using kubectl**. This will automatically set up Flyte sandbox on your AWS EKS cluster.
    
    >kubectl create -f https://raw.githubusercontent.com/flyteorg/flyte/master/deployment/sandbox/flyte_generated.yaml

5. **Port-forward**
   
    >kubectl port-forward -n projectcontour svc/envoy 30081:80

    ### **Open new Git Bash terminal**

6.	**Activate Python Virtual Environment**.
    
    >source ~/venv/Scripts/activate
    
7. **Upload created image to Flyte backend**

    >cd ../myflyteapp

    >pyflyte --pkgs myapp.workflows package --image \<registry/repo:version> -f

    >flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz --version \<version>

8. **Execute a workflow using dashboard or the command line**.
    To use dashboard simply go to http://localhost:30081/console in your web browser.
    Alternatively, you can use command line to launch and monitor a workflow.

    >flytectl get launchplan --project flytesnacks --domain development myapp.workflows.example.my_wf --latest --execFile exec_spec.yaml

    The *myapp.workflows.example.my_wf* can be changed to *myapp.workflows.\<file-name>.\<workflow-name>*

    >flytectl create execution --project flytesnacks --domain development --execFile exec_spec.yaml

    Use *flytectl get execution* to monitor the status of your workflow

    >flytectl get execution --project flytesnacks --domain development \<execname>

9. **(Optional)Build and deploy changed workflows faster**. If you want to change a workflow deployed on a cluster(and these change don't update dependencies in your requirements file), you can avoid re-building the entire Docker container. To do so, use flag *--fast* during packaging.

    > pyflyte --pkgs myapp.workflows package --image \<registry/repo:version> --fast -f
    
    After that, register changed workflow normally.

    >flytectl register files --project flytesnacks --domain development --archive flyte-package.tgz  --version \<version>

10.	**Destroy cluster**. Don’t forget to destroy created AWS EKS cluster to avoid unexpected charges.

    ### **In the first git Bash terminal**
    >^C

    >terraform destroy -auto-approve
