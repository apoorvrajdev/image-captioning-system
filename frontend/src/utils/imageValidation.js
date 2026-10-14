// Upload rules shared by the upload zone and the sample gallery, so an example
// passes exactly the checks an upload does. WebP isn't accepted: the backend
// refuses it, because its decoder can't read it.
export const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
export const MAX_BYTES = 10 * 1024 * 1024;

export const validateImageFile = (file) => {
  if (!file) return "No file selected.";
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return "Unsupported format. Use JPG or PNG.";
  }
  if (file.size > MAX_BYTES) {
    return "File exceeds 10 MB limit.";
  }
  return null;
};
