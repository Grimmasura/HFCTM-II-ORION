# TPU Setup

To run the ORION models on Google Cloud TPUs you must provide access to a TPU
node and the associated credentials.

## Credentials

1. Create a service account with access to the TPU API.
2. Download its JSON key and set the path via the standard Google environment
   variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

## Environment variables

The runtime uses the following variables to locate the TPU resource:

* `TPU_NAME` – name of the TPU node (for example, `my-tpu`).
* `TPU_ZONE` – Google Cloud zone where the TPU is hosted (for example,
  `us-central1-b`).

These values are typically provided when creating the TPU via `gcloud`:

```bash
export TPU_NAME=my-tpu
export TPU_ZONE=us-central1-b
```

With the credentials and variables set, the project will automatically
initialise the TPU using `torch_xla` or JAX depending on the available
framework.
