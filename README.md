# **Face Generation with Variational Autoencoders (VAEs)**

---

## **1. Objective**

The primary objective of this task is to **demonstrate the generative capabilities of Variational Autoencoders (VAEs)** by generating **10 diverse yet thematically consistent facial images** from a given facial image.
The system should leverage the learned latent space to create realistic variations without preserving the exact identity of the original face.

---

## **2. Introduction**

Variational Autoencoders (VAEs) are powerful **deep generative models** that learn a **continuous latent representation** of the data distribution.
In this project, we utilize a **pre-trained VAE trained on the LFW dataset** to explore its ability to synthesize new, unseen faces.

The experiment focuses on:

* Understanding the **encoding-decoding pipeline**.
* Sampling and manipulating **latent vectors (z)** to produce novel, realistic faces.
* Ensuring **diversity** while staying within the same facial domain.

---

## **3. Dataset Selection**

### **Chosen Dataset**

* **LFW (Labeled Faces in the Wild)** – Deep Funneled version.
* It contains **aligned, high-quality facial images** suitable for generative modeling.

### **Why LFW?**

* Large number of diverse faces across gender, ethnicity, and age groups.
* Widely used benchmark in face recognition and generation tasks.

### **Preprocessing Steps**

1. **Loading and Filtering**: Removed identities with too many images (>25) to balance the dataset.
2. **Resizing & Normalization**: Cropped to `45×45×3` and normalized pixel values to `[0,1]`.
3. **Train/Validation Split**: `80% training` / `20% validation` using stratified random sampling.

---

## **4. Model Architecture**

### **Variational Autoencoder (VAE)**

The implemented VAE consists of:

1. **Encoder**

   * Fully connected layers compress the image into a **latent distribution** parameterized by:

     * **μ (mu)** – Mean of the latent distribution.
     * **logσ² (logvar)** – Log variance.

2. **Reparameterization Trick**

   * Sampling latent vectors using:
     $z = \mu + \sigma \times \epsilon$,
     where $\epsilon \sim \mathcal{N}(0,1)$.

3. **Decoder**

   * Maps sampled latent vectors back to the image space using fully connected layers and a sigmoid activation.

### **Loss Function**

* **Binary Cross-Entropy (BCE)** → Measures reconstruction accuracy.
* **KL Divergence (KLD)** → Regularizes latent space to follow a standard Gaussian $\mathcal{N}(0,1)$.

$$
\mathcal{L}_{VAE} = BCE + KLD
$$

---

## **5. Training Setup**

* **Optimizer**: Adam (`lr=0.001`)
* **Epochs**: 50
* **Batch Size**: 128
* **Device**: Tesla T4 GPU (via Colab)

### **Learning Curve**

The VAE was trained until **train/validation losses converged**, confirming **no overfitting**.

examples/train_val_loss.png

---

## **6. Experimental Results**

### **Reconstruction Check**

The model successfully reconstructed validation images, confirming that the latent space captures meaningful facial features.

examples/reconstructed.png

---

### **10 Generated Images**

Using random sampling from the latent space:

1. Sampled $z \sim \mathcal{N}(0,1)$ for 10 vectors.
2. Decoded each latent vector to image space.

---

**Generated Results (2×5 Grid):**

examples/generated_images.png

* The generated faces are:

  * **Realistic** and **domain-consistent** (face-like features preserved).
  * **Diverse** (different identities, poses, and expressions).

---

## **7. Conclusions**

* **Strengths**:

  * Successfully learned a smooth, continuous latent space.
  * Able to synthesize diverse yet realistic faces from random samples.

* **Limitations**:

  * Limited resolution (45×45); fine facial details are not sharp.
  * Generated images sometimes lack clear facial attributes due to dataset bias.

* **Future Work**:

  * Train on **higher-resolution images**.
  * Experiment with **conditional VAEs (CVAE)** for controlled attribute generation.

---

## **8. Deliverables**

1. **10 Generated Images** – included above.
2. **Code Repository** – Full training and generation code.
3. **Report** – This document (Markdown/PDF).

