# HFCTM-II Production Deployment

This guide walks through deploying the HFCTM-II service to a Kubernetes cluster.

## Prerequisites

- A Kubernetes cluster with access to `majorana1` QPU and `ironwood-tpu` nodes.
- The [`azure-credentials`](#creating-required-secrets) and `google-credentials` secrets containing service credentials.
- `kubectl` configured to access the target cluster.

## Creating Required Secrets

Create the Azure and Google secrets before deploying:

```bash
kubectl create secret generic azure-credentials \
  --from-literal=client-id=<your-azure-client-id> \
  --from-literal=client-secret=<your-azure-client-secret>

kubectl create secret generic google-credentials \
  --from-file=service-account.json=<path-to-google-service-account-json>
```

## Apply the Deployment

The manifest includes a `ConfigMap`, `Deployment`, `Service`, `HPA`, `NetworkPolicy`, and `ServiceMonitor`.
Deploy everything with:

```bash
kubectl apply -f deployment/hfctm-ii-production.yaml
```

The deployment uses node selectors and tolerations for `majorana1` QPU and `ironwood-tpu` nodes to ensure pods schedule on the correct hardware.

## Verifying

Check the status of the deployment and service:

```bash
kubectl get pods,svc,hpa,netpol,servicemonitor -l app=hfctm-ii
```

Metrics are scraped via the `ServiceMonitor`, so ensure Prometheus is configured in the cluster.
