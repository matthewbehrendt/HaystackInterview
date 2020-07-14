import { Component, OnInit} from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AuthService } from '../Users/auth.service';
import { User } from '../user';
import { NotificationsService } from '../Utils/notifications.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-customer-form',
  templateUrl: './customer-form.component.html',
  styleUrls: ['./customer-form.component.css']
})
export class CustomerFormComponent implements OnInit {
  constructor(
    public http: HttpClient,
    public rout: Router,
    public authService: AuthService,
    public notificationsService: NotificationsService) {
    this.authService.getTokens();
  }

  // Initializing new user properties
  name = '';
  username = '';
  birthday = '';
  email = '';
  orgID = -1;
  Admin = false;
  notification = '';

  model: User;

  orgNames = [];
  orgIDs = [];

  submitted = false;
  valid = false;
  newOrgID = -1;
  newOrgName = '';
  newOrgCity = '';
  newOrgState = '';
  newOrgPhone = '';

  // When the page is initialized or reset, get the list of organizations to fill or update the org selection field
  ngOnInit() {

    // If the user is attempting to access this page without authorization, send them to the login page
    if (!this.authService.accessToken || !this.authService.idToken) {
      this.rout.navigate(['login']);
    }
    // Otherwise, if the user is an STC Admin, get the org names and IDs and fill the org selection field
    //  If the user is simply an org admin, they will not have the option to select an organization
    else if (this.authService.group === 'STCAdmin') {
      let resp;
      this.http.get('https://ay8oa5429i.execute-api.us-west-2.amazonaws.com/dev/organizations/all')
      .subscribe(
        response => {
          resp = response;
          for (const key in response) {
            this.orgIDs.push(key);
            this.orgNames.push(response[key].name);
          }
          do {
          this.newOrgID = Math.floor(Math.random() * (10000)) + 1;
          } while (this.orgIDs.includes(this.newOrgID));

          let options = '';
          options += '<option value=-1></option>';
          for (let i = 0; i < this.orgNames.length; i++) {
            options += '<option value="' + this.orgIDs[i] + '">' + this.orgNames[i] + '</option>';
          }
          document.getElementById('OrgNames').innerHTML = options;
        }
      );
    }
  }

  // When the form is submitted, check that the fields are valid and add the new user
  onSubmit(validEmail : boolean, elem) {
    const creation_button = (<HTMLButtonElement>document.getElementById('create'));
    
    // Check if a new customer admin is being created by STC, in which case a new organization is being created as well
    let admingroup;
    if (this.Admin) {
      admingroup = 'customerAdmin';
      this.orgID = this.newOrgID;
    } else {
      admingroup = 'customer';
    }

    // Check that all of the fields are valid
    let validity = (this.name != '' && this.username != '' && this.birthday.length > 0 && validEmail &&
    ((this.authService.group === 'STCAdmin' && 
      (!this.Admin && this.orgID != -1) || /* A new user is being created under an existing org */
      (this.Admin && this.newOrgName != '' && this.newOrgCity != '' && this.newOrgState != '' && this.newOrgPhone != '')) /* A new customer Admin is being created with a new org */ || 
    (this.authService.group !== 'STCAdmin') /* A new user is being created under the org of the logged in customer admin */ ));
    
    // Used to keep track of the submission of the form, activating the alerts for any invalid fields
    this.submitted = true;
    
    // If the form is valid, disable the form while it is being processed and add the new user
    if (validity && !elem.disabled) {
      elem.disabled = true;
      this.valid = true;
      let orgName;

      creation_button.textContent = 'Processing';
      creation_button.disabled = true;

      if (this.Admin) {
        orgName = this.newOrgName;
      } 
      else { 
        // Find the index of the selected orgID and use this to set the orgName
        let index = this.orgIDs.findIndex(x => x == this.orgID);
        orgName = this.orgNames[index];
      }
      
      // Set new user object properties based on field values
      let obj;
      if (this.Admin) {
          obj = { name: this.name,
            birthday: this.birthday.replace(/\-/gi, '/'),
            email: this.email,
            username: this.username,
            orgID: +this.orgID,
            organization: orgName,
            groupName: admingroup,
            headquarters: [this.newOrgCity, this.newOrgState],
            contactInfo: this.newOrgPhone
          };
      } else {
          obj = { name: this.name,
          birthday: this.birthday.replace(/\-/gi, '/'),
          email: this.email,
          username: this.username,
          orgID: +this.orgID,
          organization: orgName,
          groupName: admingroup
        };
      }

      // Convert the object to a JSON string and send it in a post request
      const parameter = JSON.stringify(obj);
      this.http.post('https://ay8oa5429i.execute-api.us-west-2.amazonaws.com/dev/user', parameter)
      .subscribe(data => {
        // Notify the user either that there has been an error or that the user was created successfully
        if (data['errorMessage']) {
          this.notificationsService.message = data['errorMessage'];
          this.notificationsService.visible = true;
          this.notificationsService.err = true;
        } else {
          this.notificationsService.message = 'Successfully created new User';
          this.notificationsService.visible = true;
          this.notificationsService.err = false;
          // If successful, reset the form
          this.ngOnInit();
        }
        elem.disabled = false;
        creation_button.disabled = false;
        creation_button.textContent = 'Submit';
      }, error => {
        console.log(error);
        elem.disabled = false;
        creation_button.disabled = false;
        creation_button.textContent = 'Submit';
      });
    }
  }
}
