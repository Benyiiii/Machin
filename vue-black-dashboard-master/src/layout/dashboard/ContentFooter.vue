<template>
  <footer class="footer">
    <div class="container-fluid">
      <ul class="nav">
        <li class="nav-item">
          <a class="nav-link" href="https://github.com/subrosa-w/EDA_FundamentosML_BenjaminSandoval">
            Github
          </a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="https://drive.google.com/drive/folders/1pjS7nwuUO46Se3rznEf2p9gAWkCkFHMB?usp=drive_link"> Drive </a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="http://www.creative-tim.com/license">
            Licenses
          </a>
        </li>
      </ul>
      <div class="copyright">
        © {{ year }} Hecho por Benjamin Sandoval | Benjamin Oviedo | Felipe Olavarria 
      </div>
    </div>
  </footer>
</template>
<script>
export default {
  data() {
    return {
      year: new Date().getFullYear(),
    };
  },
};
</script>
<style></style>
