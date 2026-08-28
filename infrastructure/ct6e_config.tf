# This code is compatible with Terraform 4.25.0 and versions that are backwards compatible to 4.25.0.
# For information about validating this Terraform code, see https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

resource "google_compute_instance" "instance-20260826-20260826-221331" {
  boot_disk {
    auto_delete = true
    device_name = "instance-20260826-20260826-221331"

    initialize_params {
      image = "projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20260826"
      size  = 10
      type  = "hyperdisk-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  guest_accelerator {
    count = 1
    type  = "projects/propane-facet-351921/zones/us-central1-b/acceleratorTypes/ct6e"
  }

  labels = {
    goog-ec-src           = "vm_add-tf"
    goog-ops-agent-policy = "v2-template-1-7-0"
  }

  machine_type = "ct6e-standard-1t"

  metadata = {
    accelerator-type         = "v6e-1"
    agent-worker-number      = "0"
    enable-osconfig          = "TRUE"
    tpu-env                  = "TPU_WORKER_ID: '0'\nCONSUMER_PROJECT_ID: 'propane-facet-351921'\nZONE: 'us-central1-b'\nNODE_ID: '0'\nTPU_RUNTIME_METRICS_PORTS: '8431,8432,8433,8434,8435,8436,8437,8438'\nENABLE_ICI_RESILIENCY: 'true'\nENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY: 'false'\nINJECT_SLICE_BUILDER_FAULT: ''\nTPU_ACCELERATOR_TYPE: 'v6e-1'\nTOPOLOGY: '1x1'\nWRAP: 'false,false,false'\nALT: 'false'\nHOST_BOUNDS: '1,1,1'\nCHIPS_PER_HOST_BOUNDS: '1,1,1'\n"
    worker-network-endpoints = "instance-20260826-215213"
  }

  name = "instance-20260826-20260826-221331"

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/propane-facet-351921/regions/us-central1/subnetworks/default"
  }

  reservation_affinity {
    type = "NO_RESERVATION"
  }

  scheduling {
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
    preemptible         = false
    provisioning_model  = "SPOT"
  }

  service_account {
    email  = "766744746389-compute@developer.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/devstorage.read_only", "https://www.googleapis.com/auth/logging.write", "https://www.googleapis.com/auth/monitoring.write", "https://www.googleapis.com/auth/service.management.readonly", "https://www.googleapis.com/auth/servicecontrol", "https://www.googleapis.com/auth/trace.append"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  zone = "us-central1-b"
}

module "ops_agent_policy" {
  source          = "github.com/terraform-google-modules/terraform-google-cloud-operations/modules/ops-agent-policy"
  project         = "propane-facet-351921"
  zone            = "us-central1-b"
  assignment_id   = "goog-ops-agent-v2-template-1-7-0-us-central1-b"
  agents_rule = {
    package_state = "installed"
    version = "latest"
  }
  instance_filter = {
    all = false
    inclusion_labels = [{
      labels = {
        goog-ops-agent-policy = "v2-template-1-7-0"
      }
    }]
  }
}

