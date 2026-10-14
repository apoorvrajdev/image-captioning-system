// Example images for the "Try an example" gallery (TASK-033).
//
// All five come from the committed COCO 2017 evaluation slice and are served
// unmodified (SHA-256 below). Each is CC BY 2.0: COCO's official image licence
// record and the photo's Flickr page agreed when checked on 2026-10-14.
//
// `reference` is the first reference caption stored for the image in the
// evaluation slice, in COCO's original wording. It is a dataset annotation,
// never the model's output.

const CC_BY_2 = {
  name: "CC BY 2.0",
  url: "https://creativecommons.org/licenses/by/2.0/",
};

export const REFERENCE_CAPTION_CREDIT = {
  source: "COCO Consortium",
  sourceUrl: "https://cocodataset.org/",
  licence: {
    name: "CC BY 4.0",
    url: "https://creativecommons.org/licenses/by/4.0/",
  },
};

const DATASET = "COCO 2017 train2017, evaluation slice";
const asset = (fileName) => `${import.meta.env.BASE_URL}samples/${fileName}`;

export const SAMPLES = [
  {
    id: "kitten-keyboard",
    label: "Kitten on a computer keyboard",
    fileName: "000000418799.jpg",
    src: asset("000000418799.jpg"),
    dataset: DATASET,
    cocoId: 418799,
    sha256: "b1d8cae66ae3ab40645c6644a85c100000bffae4906ad3d3dda94a3ba4d98fe1",
    credit: {
      title: "Mango HaX0ring",
      creator: "Slava",
      sourceUrl: "https://www.flickr.com/photos/slava/496607907/",
      licence: CC_BY_2,
    },
    reference: "A kitten sitting on the keyboard of a computer.",
  },
  {
    id: "fire-hydrant",
    label: "Red fire hydrant on a grass verge",
    fileName: "000000316059.jpg",
    src: asset("000000316059.jpg"),
    dataset: DATASET,
    cocoId: 316059,
    sha256: "bbeec9f2f40219d5db06cb15169a9dd529be37742ea0d7d7bcd9cd1e6e237537",
    credit: {
      title: "07252008",
      creator: "pa1nt",
      sourceUrl: "https://www.flickr.com/photos/pa1nt/2707331770/",
      licence: CC_BY_2,
    },
    reference: "A red fire hydrant sitting on the side of a road.",
  },
  {
    id: "living-room",
    label: "Living room with a sofa and fireplace",
    fileName: "000000215132.jpg",
    src: asset("000000215132.jpg"),
    dataset: DATASET,
    cocoId: 215132,
    sha256: "0fc939ede28eaba1cc11517ca2a800af550926becc52e508aa652fb8328d9b1f",
    credit: {
      title: "Living room w/painted cable covers",
      creator: "Amy West",
      sourceUrl: "https://www.flickr.com/photos/amy_elizabeth_west/5067711955/",
      licence: CC_BY_2,
    },
    reference:
      "a small living room with a blue couch, flat screen television and a fireplace",
  },
  {
    id: "propeller-aircraft",
    label: "Propeller aircraft in a blue sky",
    fileName: "000000422807.jpg",
    src: asset("000000422807.jpg"),
    dataset: DATASET,
    cocoId: 422807,
    sha256: "a6c5d6cd76a9c899e0dbdd2262b82bc84a1c426249fb27ca63f2faade13d42d4",
    credit: {
      title: "US Coast Guard Herc",
      creator: "Simon_sees",
      sourceUrl: "https://www.flickr.com/photos/39551170@N02/6815669873/",
      licence: CC_BY_2,
    },
    reference: "An airplane flying through a blue sky with propellers.",
  },
  {
    id: "fruit-platter",
    label: "Platter of sliced fruit",
    fileName: "000000302885.jpg",
    src: asset("000000302885.jpg"),
    dataset: DATASET,
    cocoId: 302885,
    sha256: "3551fe50ea9c5a2c82012d0b7aa298042301db2614d6affbac0edf50d0d488b8",
    credit: {
      title: "fruit platter",
      creator: ">littleyiye<",
      sourceUrl: "https://www.flickr.com/photos/little_yiye/3742500176/",
      licence: CC_BY_2,
    },
    reference: "A collection of fruits are stacked on a plate.",
  },
];
