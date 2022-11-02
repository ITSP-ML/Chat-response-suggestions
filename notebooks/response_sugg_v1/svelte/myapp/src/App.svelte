<script>
  import AutoComplete from "simple-svelte-autocomplete";
  let selectedCountry;

  // async function f(keyword) {
  //   const myImage = document.querySelector(".my-image");
  //   await fetch(
  //     "https://upload.wikimedia.org/wikipedia/commons/7/77/Delete_key1.jpg"
  //   )
  //     .then((response) => {
  //       console.log(response.bodyUsed);
  //       const res = response.blob();
  //       console.log(response.bodyUsed);
  //       return res;
  //     })
  //     .then((response) => {
  //       const objectURL = URL.createObjectURL(response);
  //       // myImage.src = objectURL;
  //     });
  // }

  async function get_suggestions_list(keyword) {
    // let x = JSON.stringify({ text: "P" });
    let data = { text: keyword };

    // const url2 =
    //   "https://restcountries.com/v2/name/" +
    //   encodeURIComponent(keyword) +
    //   "?fields=name;alpha2Code";
    // const url = "http://127.0.0.1:8000?text=" + encodeURIComponent(keyword);
    const url = "http://127.0.0.1:8000/";
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    return response.json();
  }
</script>

<AutoComplete
  searchFunction={get_suggestions_list}
  bind:selectedItem={selectedCountry}
  labelFieldName="sugg"
  maxItemsToShowInList={10}
  delay={200}
  localFiltering={false}
/>
