<a name="lambda-ssh-key"></a>

# ![Lambda SSH key](../assets/docs/headings/h1-lambda-ssh-key.svg)

Use `/setup` in the harness or run `./velocicyber.sh setup` to enter the registered Lambda SSH key name and
select a private key. You can choose a numbered file from this directory or enter
a path. A key outside the installation is copied here with owner-only permissions;
its contents are never stored in `Modules.txt` or printed. Enter skips this section.
The matching public key must already be registered with Lambda.

For manual configuration, place your own private key in this directory and register its matching public key
with your Lambda account. For a key named `lambda.pem`, run from the repository root:

```bash
chmod 600 ssh-keys/lambda.pem
```

Set these values in your local `Modules.txt`:

```text
SSH_PRIVATE_KEY=ssh-keys/lambda.pem
LAMBDA_SSH_KEY_NAME=your-registered-key-name
```

The private-key path must resolve inside this installation. `LAMBDA_SSH_KEY_NAME`
is the name registered with Lambda, not the local filename. Leave `LAMBDA_REGION`
blank to choose a compatible region with capacity, or set a preferred region.

Private keys are ignored by Git. Use `/local` to select a GPU model, then `/gpu`
to review an instance and approve launch. See [Lambda model setup](../docs/models.md#lambda-gpu-setup)
for inference-tunnel and host-key behavior.
