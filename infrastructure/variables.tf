variable "region" {
  type        = string
  description = "gcloud region for TPU VM"
  default     = "us-south1"
}

variable "zone" {
    type = string
    description = "gcloud zone for TPU VM"
    default = "us-south1-a"
}