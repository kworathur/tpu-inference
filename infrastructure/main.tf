provider "google-beta" {
  project     = "propane-facet-351921"
  region      = "us-south1"
  zone = "us-south1-a"
}

data "google_tpu_v2_runtime_versions" "available" {
  provider = google-beta
}

data "google_tpu_v2_accelerator_types" "available" {
  provider = google-beta
}

resource "google_tpu_v2_vm" "tpu" {
  provider = google-beta

  name = "keshav-tpu"
  description = "Test TPU for vLLM experiments."

  runtime_version  = "v2-alpha-tpuv5-lite"

  accelerator_config {
    type     = "V5LITE_POD"
    topology = "1x1"
  }

  scheduling_config {
    preemptible = true
    spot = true
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
    scope = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  data_disks {
    source_disk = google_compute_disk.disk.id                                                      
    mode        = "READ_ONLY"
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
  region        = "us-south1"
  network       = google_compute_network.network.id
}

resource "google_compute_network" "network" {
  provider = google-beta

  name                    = "tpu-net"                                                            
  auto_create_subnetworks = false
}

resource "google_compute_firewall" "allow_ssh" {
  provider  = google-beta                                                                        
  name      = "tpu-allow-ssh"
  network   = google_compute_network.network.id                                                  
  direction = "INGRESS"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["35.235.240.0/20"]  # IAP TCP forwarding range
  target_tags   = ["foo"]              # matches the tag on your TPU
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
  type  = "pd-ssd"
  zone  = "us-south1-a"
}
# Wait after service account creation to limit eventual consistency errors.                    
resource "time_sleep" "wait_60_seconds" {
  depends_on = [google_service_account.sa]
  create_duration = "60s"
}
