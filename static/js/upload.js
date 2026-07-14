/**
 * upload.js — File Upload Handler
 * =================================
 * Handles:
 *   - Drag & drop file upload with visual feedback
 *   - Client-side file validation (type, size)
 *   - Image preview before upload
 *   - Form submission with loading overlay
 */

document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initFileInput();
    initUploadForm();
});


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB


// ---------------------------------------------------------------------------
// Dropzone (Drag & Drop)
// ---------------------------------------------------------------------------

function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');

    if (!dropzone || !fileInput) return;

    // Click to open file dialog
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag events
    ['dragenter', 'dragover'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0], fileInput);
        }
    });
}


// ---------------------------------------------------------------------------
// File Input (Browse)
// ---------------------------------------------------------------------------

function initFileInput() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput) return;

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0], fileInput);
        }
    });
}


// ---------------------------------------------------------------------------
// File Handler — Validation & Preview
// ---------------------------------------------------------------------------

function handleFile(file, fileInput) {
    // --- Validate file type ---
    if (!ALLOWED_TYPES.includes(file.type)) {
        showNotification('Invalid file type. Please upload PNG, JPG, or JPEG.', 'error');
        return;
    }

    // --- Validate file size ---
    if (file.size > MAX_FILE_SIZE) {
        showNotification('File too large. Maximum size is 16 MB.', 'error');
        return;
    }

    // --- Set file in the input (for drag & drop) ---
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    // --- Show preview ---
    showPreview(file);

    // --- Enable upload button ---
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.disabled = false;
    }

    showNotification(`File "${file.name}" ready for analysis.`, 'success');
}


// ---------------------------------------------------------------------------
// Image Preview
// ---------------------------------------------------------------------------

function showPreview(file) {
    const previewContainer = document.getElementById('filePreview');
    const previewImage = document.getElementById('previewImage');
    const previewName = document.getElementById('previewName');
    const previewSize = document.getElementById('previewSize');
    const dropzone = document.getElementById('dropzone');

    if (!previewContainer) return;

    // Read file as data URL for preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Display file info
    previewName.textContent = file.name;
    previewSize.textContent = formatFileSize(file.size);

    // Show preview, hide dropzone content
    previewContainer.style.display = 'block';

    // Setup remove button
    const removeBtn = document.getElementById('removeFile');
    if (removeBtn) {
        removeBtn.addEventListener('click', () => {
            clearFile();
        });
    }
}

function clearFile() {
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('filePreview');
    const uploadBtn = document.getElementById('uploadBtn');

    if (fileInput) fileInput.value = '';
    if (previewContainer) previewContainer.style.display = 'none';
    if (uploadBtn) uploadBtn.disabled = true;
}


// ---------------------------------------------------------------------------
// Form Submission with Loading
// ---------------------------------------------------------------------------

function initUploadForm() {
    const form = document.getElementById('uploadForm');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        const fileInput = document.getElementById('fileInput');

        // Check if file is selected
        if (!fileInput || !fileInput.files.length) {
            e.preventDefault();
            showNotification('Please select a file first.', 'error');
            return;
        }

        // Show loading overlay
        showLoading();

        // Disable submit button to prevent double-submit
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        }

        // Let the form submit normally (POST to /predict)
    });
}


// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/**
 * Format file size in human-readable format.
 * @param {number} bytes - File size in bytes.
 * @returns {string} Formatted size string.
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));

    return parseFloat((bytes / Math.pow(1024, i)).toFixed(2)) + ' ' + sizes[i];
}
