provider "google-beta" {
  project = "propane-facet-351921"
  region  = "${var.region}"
  zone    = "${var.zone}"   # v5e is available here; us-central1-c only supports v2-8 / TPU7x
}

data "google_tpu_v2_runtime_versions" "available" {
  provider = google-beta
}

data "google_tpu_v2_accelerator_types" "available" {
  provider = google-beta
}

resource "google_tpu_v2_vm" "tpu" {
  provider = google-beta

  name        = "keshav-tpu"
  description = "Test TPU for vLLM experiments."

  runtime_version = "v2-alpha-tpuv5-lite"   # v5e runtime

  accelerator_config {
    type     = "V5LITE_POD"   # v5e
    topology = "1x1"         # single chip
  }

  scheduling_config {
    reserved = false
  }

  network_config {
    can_ip_forward      = true
    enable_external_ips = true
    network             = google_compute_network.network.id
    subnetwork          = google_compute_subnetwork.subnet.id
  }

  shielded_instance_config {
    enable_secure_boot = true
  }

  service_account {
    email = google_service_account.sa.email
    scope = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }

  data_disks {
    source_disk = google_compute_disk.disk.id
  }
 
  labels = {
    foo = "bar"
  }

  metadata = {
    foo = "bar"
  }

  tags = ["foo"]

  depends_on = [time_sleep.wait_60_seconds]
}

resource "google_compute_subnetwork" "subnet" {
  provider = google-beta

  name          = "tpu-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = "${var.region}"
  network       = google_compute_network.network.id
}

resource "google_compute_network" "network" {
  provider = google-beta

  name                    = "tpu-net"
  auto_create_subnetworks = false
}

resource "google_service_account" "sa" {
  provider = google-beta

  account_id   = "tpu-sa"
  display_name = "Test TPU VM"
}

resource "google_compute_disk" "disk" {
  provider = google-beta

  name  = "tpu-disk"
  image = "debian-cloud/debian-12"
  size  = 10
  type  = "pd-balanced"
  zone  = "${var.zone}"   # matches TPU zone
}

# Wait after service account creation to limit eventual consistency errors.
resource "time_sleep" "wait_60_seconds" {
  depends_on = [google_service_account.sa]

  create_duration = "60s"
}