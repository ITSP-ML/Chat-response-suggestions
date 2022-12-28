<script>
  import AutoComplete from "simple-svelte-autocomplete";
  let selectedCountry;

  async function get_suggestions_list(keyword) {
    // let x = JSON.stringify({ text: "P" });
    let data = { text: keyword };
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
  lowercaseKeywords={true}
/>
<!-- <script>
  import AutoComplete from "simple-svelte-autocomplete";
  let selectedCountry;
  async function searchCountry(keyword) {
    const url =
      "https://restcountries.com/v2/name/" +
      encodeURIComponent(keyword) +
      "?fields=name;alpha2Code";

    const response = await fetch(url);
    return await response.json();
  }
</script>

<AutoComplete
  searchFunction={searchCountry}
  bind:selectedItem={selectedCountry}
  labelFieldName="name"
  maxItemsToShowInList={10}
  delay={200}
  localFiltering={false}
/> -->
